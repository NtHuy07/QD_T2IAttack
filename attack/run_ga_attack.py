import numpy as np
import os
import sys
import random
import shutil
from nltk.tokenize import RegexpTokenizer
from english import ENGLISH_FILTER_WORDS
from compute_img_sim import compute_img_sim
from attack_lib import attack
import argparse
from numpy.linalg import norm
from map_elites import MAPElites, CVTMAPElites
import time

from sentence_transformers import SentenceTransformer
sent_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')



def check_if_contains(tokens):
    flag = False
    loc = 0
    for token in tokens:
        if "_" in token:
            flag = True
            break
        loc += 1
    return flag, loc

def check_if_in_list(sent, sent_ls):
    flag = False
    for tar_sent in sent_ls:
        if sent == tar_sent:
            flag = True
            break
    return flag


def get_new_pop(elite_pop, elite_pop_scores, pop_size):

    elite_pop_probs = np.ones(len(elite_pop))/len(elite_pop)

    cand1 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]

    #exchange two parts randomly
    mask = np.random.rand(pop_size, len(elite_pop[0])) < 0.5 
    
    next_pop = []
    pop_index = 0
    for pop_flag in mask:
        pop = []
        word_index = 0
        for word_flag in pop_flag:
            if word_flag:
                pop.append(cand1[pop_index][word_index])
            else:
                pop.append(cand2[pop_index][word_index])
            word_index += 1
        next_pop.append(pop)
        pop_index += 1

    return next_pop
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

class Genetic():
    
    def __init__(self, ori_sent, tar_img_path, tar_sent, log_save_path, intem_img_path, 
                 elite_img_path, log_path, save_all_images, best_img_path):
        
        self.init_pop_size = 150
        self.pop_size = 10
        self.elite_size = 8
        self.mutation_p = 0.8
        self.mu = 0.99
        self.alpha = 0.001
        self.max_iters = 100
        self.store_thres = 80

        self.target_img_path = tar_img_path
        self.log_save_path = log_save_path
        self.intermediate_path = intem_img_path
        self.best_img_path = best_img_path
        self.elite_img_path = elite_img_path
        self.save_all_images = save_all_images
        
        self.ori_sent = ori_sent
        self.target_sent = tar_sent

        self.ori_enc = sent_encoder.encode(self.ori_sent)
        self.tar_enc = sent_encoder.encode(self.target_sent)

        map_grid_size = (10,)
        behavior_space = np.array([[10, 10], [90, 60]])

        #initialize MAP-Elites
        self.me = MAPElites(map_grid_size=map_grid_size, 
                            behavior_space=behavior_space, 
                            log_snapshot_dir=os.path.join(log_path, "iterations"), 
                            history_length=1, 
                            seed=42)
        
        #initialize attack class
        self.attack_cls = attack(self.target_sent)
        
        #initialize tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        tokens = self.tokenizer.tokenize(ori_sent.lower())     

        self.max_sent = None
        self.max_fitness = -np.inf

        #generate large initialization corpus
        self.pop = self.initial_mutate(tokens, self.init_pop_size)
        print("initial pop: ", self.pop)

    def initial_mutate(self, pop, nums):
        #random select the pop sentence that will mutate 
        new_pop = [pop]
        new_sent_ls = [" ".join(pop)]
        
        #filter the original sentence
        legal_token = []
        for k, tok in enumerate(pop):
            if tok.lower() not in ENGLISH_FILTER_WORDS:
                legal_token.append(k)


        #append the list until it fills out nums
        count = 0
        repeat_cnt = 0
        while count < nums-1:
            word_idx = legal_token[np.random.choice(len(legal_token), size=1)[0]]
            word = pop[word_idx]

            bug = self.attack_cls.selectBug(word, pop, word_idx, None)
            # tokens = self.attack_cls.replaceWithBug(pop, word_idx[0], bug)
            tokens = self.attack_cls.replaceWithBug(pop, word_idx, bug)
            #join it into a sentence
            x_prime_sent = " ".join(tokens)
            if (check_if_in_list(x_prime_sent, new_sent_ls)) and repeat_cnt < 10: # avoid iterating permanently
                repeat_cnt += 1
                continue
            repeat_cnt = 0

            new_sent_ls.append(x_prime_sent)
            new_pop.append(tokens)
            count += 1
            print("current count: ", count)
        
        x_prime_sent_fitness = self.get_fitness_score(new_pop, 0)
        x_prime_sent_pheno = self.get_phenotype(new_sent_ls)
        self.me.update_map(new_pop, x_prime_sent_fitness, x_prime_sent_pheno, 
                                             self.max_sent, self.max_fitness, 0)

        return new_pop


    def get_phenotype(self, inputs):
        inputs_enc = sent_encoder.encode(inputs)

        ori_sim = 100 * np.dot(inputs_enc, self.ori_enc)/(norm(inputs_enc, axis=-1)*norm(self.ori_enc))
        tar_sim = 100 * np.dot(inputs_enc, self.tar_enc)/(norm(inputs_enc, axis=-1)*norm(self.tar_enc))

        return np.array([ori_sim, tar_sim]).T
    

    def get_fitness_score(self, input_tokens, gen):
        #get fitness score of all the sentences
        sim_score_ls = []

        for cnt, tokens in enumerate(input_tokens):
            x_prime_sent = " ".join(tokens)
            x_prime_sent = x_prime_sent.replace("_", " ")

            
            x_img_path = self.intermediate_path + "gen.png"
            
            gen_img_from_text(x_prime_sent, x_img_path)

            similarity = compute_img_sim(x_img_path, self.target_img_path)

            # save image
            if self.save_all_images:
                os.makedirs(self.elite_img_path + f"/itr_{str(gen)}", exist_ok=True)
                elite_path = self.elite_img_path + f"/itr_{str(gen)}/{str(cnt)}.png"
                shutil.copy(x_img_path, elite_path)
            
            sim_score_ls.append(similarity.item())

            print(f"x_prime_sent: {x_prime_sent}, similarity: {similarity.item()}")
        sim_score_arr = np.array(sim_score_ls)
        return sim_score_arr
    
    def mutate_pop(self, pop, mutation_p):
        #random select the pop sentence that will mutate
        mask = np.random.rand(len(pop)) < mutation_p 
        new_pop = []
        pop_index = 0
        for flag in mask:
            if not flag:
                new_pop.append(pop[pop_index])
            else:
                tokens = pop[pop_index]

                legal_token = []
                for k, tok in enumerate(tokens):
                    if tok.lower() not in ENGLISH_FILTER_WORDS:
                        legal_token.append(k)
                
                word_idx = legal_token[np.random.choice(len(legal_token), size=1)[0]]
                word = tokens[word_idx]

                word_slice = word.split("_")
                if len(word_slice) > 1:
                    #randomly choose one
                    sub_word_idx = np.random.choice(len(word_slice), size=1)
                    sub_word = word_slice[sub_word_idx[0]]
                    bug = self.attack_cls.selectBug(sub_word, tokens, word_idx, sub_word_idx[0])
                    word_slice[sub_word_idx[0]] = bug
                    final_bug = '_'.join(word_slice)
                else:
                    final_bug = self.attack_cls.selectBug(word, tokens, word_idx, None)

                tokens = self.attack_cls.replaceWithBug(tokens, word_idx, final_bug)
                new_pop.append(tokens)
            pop_index += 1
        
        return new_pop
                    
    def run(self, log=None):
        best_save_dir = self.best_img_path
        elite_save_dir = self.elite_img_path
        itr = 1
        prev_score = None
        save_dir = self.intermediate_path
        best_score = float("-inf")
        if log is not None:
            log.write('target phrase: ' + self.target_sent + '\n')
        
        while itr <= self.max_iters:
            
            print(f"-----------itr num:{itr}----------------")
            log.write("------------- iteration:" + str(itr) + " ---------------\n")
            print("Max fitness: ", self.me.max_fitness(), " Coverage: ", self.me.coverage(), 
                  " Niches filled: ", self.me.niches_filled(), " QD Score: ", self.me.qd_score())

            pop_sents = []
            for tokens in self.pop:
                x_prime_sent = " ".join(tokens)
                x_prime_sent = x_prime_sent.replace("_", " ")
                pop_sents.append(x_prime_sent)

            pop_scores = self.get_fitness_score(self.pop, itr)
            pop_phenos = self.get_phenotype(pop_sents)

            #store to repertoire for computation
            self.me.update_map(self.pop, pop_scores, pop_phenos, 
                            self.max_sent, self.max_fitness, itr)

            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop = [self.pop[i] for i in elite_ind]
            elite_pop_scores = pop_scores[elite_ind]

            print("current best score: ", elite_pop_scores[-1])
            
            for i in elite_ind:
                if pop_scores[i] > self.store_thres:
                    x_prime_sent_store = " ".join(self.pop[i])
                    x_prime_sent_store = x_prime_sent_store.replace("_", " ")
                    log.write(str(pop_scores[i]) + " " + x_prime_sent_store + "\n")
            
            if elite_pop_scores[-1] > best_score:
                best_score = elite_pop_scores[-1]
                #store the current best image
                x_prime_sent = " ".join(elite_pop[-1])
                x_prime_sent = x_prime_sent.replace("_", " ")
                
                x_img_path = save_dir + "gen.png"

                gen_img_from_text(x_prime_sent, x_img_path)

                best_ori_path = best_save_dir + "itr_" + str(itr) + "_score_" + str(elite_pop_scores[-1]) + ".png"
                shutil.copy(x_img_path, best_ori_path)

                #new best adversarial sentences
                log.write("new best adv: " +  str(elite_pop_scores[-1]) + " " + x_prime_sent + "\n")
                log.flush()

            self.me.save_results(itr)
       
            if prev_score is not None and prev_score != elite_pop_scores[-1]: 
                self.mutation_p = self.mu * self.mutation_p + self.alpha / np.abs(elite_pop_scores[-1] - prev_score) 
            
            next_pop = get_new_pop(elite_pop, elite_pop_scores, self.pop_size)

            self.pop = self.mutate_pop(next_pop, self.mutation_p)

            prev_score = elite_pop_scores[-1]
            self.me.visualize(itr)
            itr += 1

        return 
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_sent', type=str, required=True, help='original sentence')
    parser.add_argument('--tar_img_path', type=str, required=True, help='target image path')
    parser.add_argument('--tar_sent', type=str, required=True, help='target sentence')
    parser.add_argument('--tar_model', type=str, required=True, default='dalle-mini', 
                        choices=['dalle-mini', 'imagen', 'sd3'], help='target model')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the root path to save logs')
    parser.add_argument('--log_save_path', type=str, default='run_log.txt', help='path to save log')
    parser.add_argument('--intem_img_path', type=str, default='./intermediate_img_path/', help='path to save intermediate imgs')
    parser.add_argument('--best_img_path', type=str, default='./best_img_path/', help='path to save best output imgs')
    parser.add_argument('--elite_img_path', type=str, default='./elite_img_path/', help='path to save elite output imgs')
    parser.add_argument('--save_all_images', action='store_true', help='save every generated image during search')
    args = parser.parse_args()

    sys.path.append('../target_model')
    if args.tar_model == 'dalle-mini':
        from image_from_text_dalle import gen_img_from_text
    elif args.tar_model == 'imagen':
        from image_from_text_imagen import gen_img_from_text
    elif args.tar_model == 'sd3':
        from image_from_text_sd3 import gen_img_from_text
    else:
        raise ValueError('Target model is not supported')
    
    args.log_save_path = os.path.join(args.log_path, args.log_save_path)
    args.best_img_path = os.path.join(args.log_path, args.best_img_path)
    args.elite_img_path = os.path.join(args.log_path, args.elite_img_path)

    os.makedirs(args.best_img_path, exist_ok=True)
    os.makedirs(args.elite_img_path, exist_ok=True)
    os.makedirs(args.intem_img_path, exist_ok=True)

    st = time.time()
    g = Genetic(args.ori_sent, args.tar_img_path, args.tar_sent, args.log_save_path, args.intem_img_path, 
                args.elite_img_path, args.log_path, args.save_all_images, args.best_img_path)
    with open(args.log_save_path, 'w') as log:
        g.run(log=log)
    print(time.time() - st)
