import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]


class Map:

    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        self.history_length: int = history_length
        self.dims: tuple = dims
        if self.history_length == 1:
            self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        else:
            # Set starting top of buffer to 0 (% operator)
            self.top = np.full(dims, self.history_length - 1, dtype=int)
            self.array = np.full(
                (history_length,) + tuple(dims), fill_value, dtype=dtype
            )
        self.empty = True

    def __getitem__(self, map_ix):
        if self.history_length == 1:
            return self.array[map_ix]
        else:
            return self.array[(self.top[map_ix], *map_ix)]

    def __setitem__(self, map_ix, value):
        self.empty = False
        if self.history_length == 1:
            self.array[map_ix] = value
        else:
            top_val = self.top[map_ix]
            top_val = (top_val + 1) % self.history_length
            self.top[map_ix] = top_val
            self.array[(self.top[map_ix], *map_ix)] = value

    def assign_fitness_in_depth(self, map_ix, value: float) -> int:
        indices_at_bin = (slice(None),) + map_ix
        # expecting a non-empty index, only calling this method when we know
        # current fitness can be placed somewhere
        insert_idx = np.where(self.array[indices_at_bin] < value)[0][-1]
        new_bin_fitnesses = np.concatenate(
            (
                self.array[indices_at_bin][1 : insert_idx + 1],
                np.array([value]),
                self.array[indices_at_bin][insert_idx + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_fitnesses
        return insert_idx

    def insert_individual_at_depth(self, map_ix, depth, individual):
        indices_at_bin = (slice(None),) + map_ix
        new_bin_individuals = np.concatenate(
            (
                self.array[indices_at_bin][1 : depth + 1],
                np.array([individual]),
                self.array[indices_at_bin][depth + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_individuals

    @property
    def latest(self) -> np.ndarray:
        if self.history_length == 1:
            return self.array
        else:
            # should be equivalent to np.choose(self.top, self.array), but without limit of 32 choices
            return np.take_along_axis(
                arr=self.array, indices=self.top[np.newaxis, ...], axis=0
            ).squeeze(axis=0)

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def map_size(self) -> int:
        if self.history_length == 1:
            return self.array.size
        else:
            return self.array[0].size

    @property
    def qd_score(self) -> float:
        return self.latest[np.isfinite(self.latest)].sum()

    @property
    def max(self) -> float:
        return self.latest.max()

    @property
    def min(self) -> float:
        return self.latest.min()

    @property
    def max_finite(self) -> float:
        if not np.isfinite(self.latest).any():
            return np.NaN
        else:
            return self.latest[np.isfinite(self.latest)].max()

    @property
    def min_finite(self) -> float:
        if not np.isfinite(self.latest).any():
            return np.NaN
        else:
            return self.latest[np.isfinite(self.latest)].min()

    @property
    def mean(self) -> float:
        return self.latest[np.isfinite(self.latest)].mean()

    @property
    def niches_filled(self) -> int:
        return np.count_nonzero(np.isfinite(self.array))


class MAPElitesBase:

    def __init__(
        self,
        log_snapshot_dir,
        history_length,
        save_history,
        save_snapshot_interval,
        save_np_rng_state,
        load_np_rng_state,
        seed,
        init_map: Optional[Map] = None,
    ):

        self.output_dir = log_snapshot_dir
        self.history_length = history_length
        self.save_history = save_history
        self.save_snapshot_interval = save_snapshot_interval
        self.start_step = 0
        self.save_np_rng_state = save_np_rng_state
        self.load_np_rng_state = load_np_rng_state
        self.rng = np.random.default_rng(seed)
        self.rng_generators = None

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        self._init_discretization()
        self._init_maps(init_map, log_snapshot_dir)
        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self):
        raise NotImplementedError

    def _get_map_dimensions(self):
        raise NotImplementedError

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        raise NotImplementedError

    def visualize(self):
        pass

    def _init_maps(
        self, init_map: Optional[Map] = None, log_snapshot_dir: Optional[str] = None
    ):
        # perfomance of niches
        if init_map is None:
            self.map_dims = self._get_map_dimensions()
            self.fitnesses: Map = Map(
                dims=self.map_dims,
                fill_value=-np.inf,
                dtype=float,
                history_length=self.history_length,
            )
        else:
            self.map_dims = init_map.dims
            self.fitnesses = init_map

        # niches' sources
        self.genomes: Map = Map(
            dims=self.map_dims,
            fill_value=-1.0,
            dtype=object,
            history_length=self.history_length,
        )
        
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.map_dims, fill_value=False, dtype=bool)
        self.pheno: Map = Map(dims=self.map_dims, fill_value=-1.0, dtype=object)
        self.img_id: Map = Map(dims=self.map_dims, fill_value=-1.0, dtype=object)

        log_path = Path(log_snapshot_dir)
        if log_snapshot_dir and os.path.isdir(log_path):
            stem_dir = log_path.stem

            assert (
                "step_" in stem_dir
            ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
            self.start_step = (
                int(stem_dir.replace("step_", "")) + 1
            )  # add 1 to correct the iteration steps to run

            snapshot_path = log_path / "maps.pkl"
            assert os.path.isfile(
                snapshot_path
            ), f'{log_path} does not contain map snapshot "maps.pkl"'
            # first, load arrays and set them in Maps
            # Load maps from pickle file
            with open(snapshot_path, "rb") as f:
                maps = pickle.load(f)
            assert (
                self.genomes.array.shape == maps["genomes"].shape
            ), f"expected shape of map doesn't match init config settings, got {self.genomes.array.shape} and {maps['genomes'].shape}"

            self.genomes.array = maps["genomes"]
            self.fitnesses.array = maps["fitnesses"]
            self.nonzero.array = maps["nonzero"]
            self.pheno.array = maps["phenno"]
            self.img_id.array = maps["img_id"]
            # check if one of the solutions in the snapshot contains the expected genotype type for the run
            assert not np.all(
                self.nonzero.array is False
            ), "snapshot to load contains empty map"

            # compute top indices
            if hasattr(self.fitnesses, "top"):
                top_array = np.array(self.fitnesses.top)
                for cell_idx in np.ndindex(
                    self.fitnesses.array.shape[1:]
                ):  # all indices of cells in map
                    nonzero = np.nonzero(
                        self.fitnesses.array[(slice(None),) + cell_idx] != -np.inf
                    )  # check full history depth at cell
                    if len(nonzero[0]) > 0:
                        top_array[cell_idx] = nonzero[0][-1]
                # correct stats
                self.genomes.top = top_array.copy()
                self.fitnesses.top = top_array.copy()
            self.genomes.empty = False
            self.fitnesses.empty = False

            history_path = log_path / "history.pkl"
            if self.save_history and os.path.isfile(history_path):
                with open(history_path, "rb") as f:
                    self.history = pickle.load(f)
            with open((log_path / "fitness_history.pkl"), "rb") as f:
                self.fitness_history = pickle.load(f)

            print("Loading finished")

    def random_id_selection(self, num_random) -> MapIndex:
        ix = self.rng.choice(np.flatnonzero(self.nonzero.array), num_random)
        return np.unravel_index(ix, self.nonzero.dims)
    
    def random_selection(self, num_random):
        rand_id = self.random_id_selection(num_random)
        return self.genomes[rand_id], self.fitnesses[rand_id]

    def update_map(self, new_individuals, fitnesses, phenotypes, max_genome, max_fitness, gen):
        # `new_individuals` is a list of generation/mutation. We put them
        # into the behavior space one-by-one.
        for i, individual in enumerate(new_individuals):
            fitness = fitnesses[i]
            if np.isinf(fitness):
                continue
            phenotype = phenotypes[i]
            map_ix = self.to_mapindex(phenotype)

            # if the return is None, the individual is invalid and is thrown
            # into the recycle bin.
            if map_ix is None:
                self.recycled[self.recycled_count % len(self.recycled)] = individual
                self.recycled_count += 1
                continue

            if self.save_history:
                self.history[map_ix].append(individual)

            self.nonzero[map_ix] = True

            # If new fitness greater than old fitness in niche, replace.
            if fitness > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = fitness
                self.genomes[map_ix] = individual
                self.pheno[map_ix] = phenotype
                self.img_id[map_ix] = f"itr_{str(gen)}_{str(i)}"

            # update if new fitness is the highest so far.
            if fitness > max_fitness:
                max_fitness = fitness
                max_genome = individual

        return max_genome, max_fitness

    def niches_filled(self):
        return self.fitnesses.niches_filled

    def max_fitness(self):
        return self.fitnesses.max_finite

    def mean_fitness(self):
        return self.fitnesses.mean

    def min_fitness(self):
        return self.fitnesses.min_finite

    def qd_score(self):
        return self.fitnesses.qd_score
    
    def coverage(self):
        return self.niches_filled() / self.fitnesses.map_size

    def save_results(self, step: int):
        # create folder for dumping results and metadata
        output_folder = Path(self.output_dir + f"/step_{step}")
        os.makedirs(output_folder, exist_ok=True)
        maps = {
            "fitnesses": self.fitnesses.array,
            "genomes": self.genomes.array,
            "nonzero": self.nonzero.array,
            "pheno": self.pheno.array,
            "img_id": self.img_id.array,
        }
        # Save maps as pickle file
        try:
            with open((output_folder / "maps.pkl"), "wb") as f:
                pickle.dump(maps, f)
        except Exception:
            pass
        if self.save_history:
            with open((output_folder / "history.pkl"), "wb") as f:
                pickle.dump(self.history, f)

        with open((output_folder / "fitness_history.pkl"), "wb") as f:
            pickle.dump(self.fitness_history, f)

        f.close()

    def plot_fitness(self, step: int):
        import matplotlib.pyplot as plt

        save_path: str = self.output_dir + f"/step_{step}"
        plt.figure()
        plt.plot(self.fitness_history["max"], label="Max fitness")
        plt.plot(self.fitness_history["mean"], label="Mean fitness")
        plt.plot(self.fitness_history["min"], label="Min fitness")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_fitness_history.png")
        plt.close("all")

        plt.figure()
        plt.plot(self.fitness_history["qd_score"], label="QD score")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_qd_score.png")
        plt.close("all")

        plt.figure()
        plt.plot(self.fitness_history["niches_filled"], label="Niches filled")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_niches_filled.png")
        plt.close("all")

        if len(self.map_dims) > 1:
            if len(self.fitnesses.dims) == 2:
                map2d = self.fitnesses.latest
                # print(
                #     "plotted genes:",
                #     *[str(g) for g in self.genomes.latest.flatten().tolist()],
                # )
            else:
                ix = tuple(np.zeros(max(1, len(self.fitnesses.dims) - 2), int))
                map2d = self.fitnesses.latest[ix]

                # print(
                #     "plotted genes:",
                #     *[str(g) for g in self.genomes.latest[ix].flatten().tolist()],
                # )

            plt.figure()
            plt.pcolor(map2d, cmap="inferno")
            plt.savefig(f"{save_path}/MAPElites_vis.png")
        plt.close("all")

    def visualize_individuals(self):
        """Visualize the genes of the best performing solution."""
        import matplotlib.pyplot as plt

        tmp = self.genomes.array.reshape(self.genomes.shape[0], -1)

        # if we're tracking history, rows will be the history dimension
        # otherwise, just the first dimension of the map
        plt.figure()
        _, axs = plt.subplots(nrows=tmp.shape[0], ncols=tmp.shape[1])
        for genome, ax in zip(tmp.flatten(), axs.flatten()):
            # keep the border but remove the ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            try:
                genome.visualize(ax=ax)
            except AttributeError:
                pass
        save_path: str = self.output_dir
        plt.savefig(f"{save_path}/MAPElites_individuals.png")


class MAPElites(MAPElitesBase):

    def __init__(
        self,
        map_grid_size,
        behavior_space,
        log_snapshot_dir,
        history_length,
        seed,
        save_history = True,
        save_snapshot_interval = 1,
        save_np_rng_state = True,
        load_np_rng_state = True,
        **kwargs,
    ):
        self.map_grid_size = map_grid_size
        self.behavior_space = behavior_space
        self.behavior_ndim = behavior_space[0].shape[0]
        super().__init__(
            log_snapshot_dir,
            history_length,
            save_history,
            save_snapshot_interval,
            save_np_rng_state,
            load_np_rng_state,
            seed,
        )

    def _init_discretization(self):
        """Set up the discrete behaviour space for the algorithm."""
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*self.behavior_space, self.map_grid_size[0] + 1)[1:-1].T  # type: ignore

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return self.map_grid_size * self.behavior_ndim

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def visualize(self, step):
        """Visualize the map."""
        self.plot_fitness(step)


class CVTMAPElites(MAPElitesBase):

    def __init__(
        self,
        cvt_samples,
        n_niches,
        behavior_space,
        log_snapshot_dir,
        history_length,
        seed,
        save_history = True,
        save_snapshot_interval = True,
        save_np_rng_state = True,
        load_np_rng_state = False,
        **kwargs,
    ):
        
        self.cvt_samples: int = cvt_samples
        self.n_niches: int = n_niches
        self.behavior_space = behavior_space
        self.behavior_ndim = behavior_space[0].shape[0]

        super().__init__(
            log_snapshot_dir,
            history_length,
            save_history,
            save_snapshot_interval,
            save_np_rng_state,
            load_np_rng_state,
            seed,
        )

    def _init_discretization(self):
        """Discretize behaviour space using CVT."""
        # lower and upper bounds for each dimension
        low = self.behavior_space[0]
        high = self.behavior_space[1]

        points = np.zeros((self.cvt_samples, self.behavior_ndim))
        for i in range(self.behavior_ndim):
            points[:, i] = self.rng.uniform(low[i], high[i], size=self.cvt_samples)

        k_means = KMeans(init="k-means++", n_init="auto", n_clusters=self.n_niches)
        k_means.fit(points)
        self.centroids = k_means.cluster_centers_

        self.plot_centroids(points, k_means)

    def _get_map_dimensions(self):
        return (self.n_niches,)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - self.centroids, axis=1)),)
        )

    def visualize(self):
        self.plot_fitness()
        self.plot_behaviour_space()

    def plot_centroids(self, points, k_means):
        import matplotlib.pyplot as plt

        plt.figure()
        labels = k_means.labels_
        if self.behavior_ndim == 2:
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )
                plt.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    s=10,
                    marker=".",
                    color=color,
                )
        elif self.behavior_ndim >= 3:
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )
                ax.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    points[labels == i, 2],
                    s=10,
                    marker=".",
                    c=[color],
                )
        else:
            print("Not enough dimensions to plot centroids")
            return
        save_path: str = self.output_dir
        plt.savefig(f"{save_path}/MAPElites_centroids.png")

    def plot_behaviour_space(self):
        import matplotlib.pyplot as plt

        if self.behavior_ndim == 2:
            plt.figure()
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )

                # get the first two dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:2]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        plt.scatter(
                            hist[:, 0], hist[:, 1], s=10, marker=".", color=color
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        plt.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            s=10,
                            marker=".",
                            color=color,
                        )

            plt.xlim([0, self.behavior_space[1, 0]])
            plt.ylim([0, self.behavior_space[1, 1]])

        elif self.behavior_ndim >= 3:
            plt.figure()
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )

                # get the first three dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:3]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        ax.scatter(
                            hist[:, 0],
                            hist[:, 1],
                            hist[:, 2],
                            s=10,
                            marker=".",
                            c=[color],
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        ax.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            g.to_phenotype()[2],
                            s=10,
                            marker=".",
                            c=[color],
                        )

            ax.set_xlim([0, self.behavior_space[1, 0]])
            ax.set_ylim([0, self.behavior_space[1, 1]])
            ax.set_zlim([0, self.behavior_space[1, 2]])

        else:
            print("Not enough dimensions to plot behaviour space history")
            return
        save_path: str = self.output_dir
        plt.savefig(f"{save_path}/MAPElites_behaviour_history.png")