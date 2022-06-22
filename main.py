from tqdm import tqdm
from rapidfuzz import fuzz
from fastapi import FastAPI
from nltk.corpus import stopwords
import pandas as pd
from textblob import TextBlob
import re
from config import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch.nn.functional as nnf
import pickle
import nltk
from pydantic import BaseModel

class Text(BaseModel):
    content: str

try:
    nltk.download('stopwords')
except:
    pass
try:
    nltk.download('punkt')
except:
    pass
stop_words = set(stopwords.words('english'))

import warnings
warnings.filterwarnings("ignore")

class catergory_classifier():
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_WEIGHT_FOLDER, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_WEIGHT_FOLDER, use_fast=True, local_files_only=True)
        # self.model.to('cuda')
        self.mapping = MAPPING
        self.inv_mapping = {value: key for (
            key, value) in self.mapping.items()}

    def predict(self, text):
        outputs = self.model(**self.tokenizer(text, padding='max_length', max_length=512,
                                              truncation=True,
                                              add_special_tokens=True,
                                              return_token_type_ids=False,
                                              return_attention_mask=True,
                                              return_overflowing_tokens=False,
                                              return_special_tokens_mask=False, return_tensors="pt"))
        logits = outputs.logits
        pred_probability = (nnf.softmax(logits, dim=1))
        if float(max(pred_probability[0])) > 0.70:
            return np.argmax(logits.cpu().detach().numpy(), axis=1)[0]
        else:
            return np.nan


class predict_skills_class():
    def load_pickle(self, catergory):
        with open('Models/cat_skills', 'rb') as f:
            self.all_skills = pickle.load(f)
        self.listOfKeys = [
            key for (key, value) in MAPPING.items() if value == catergory][0]
        if catergory==1:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT+str(catergory)+"_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT+str(catergory)+"_svc", 'rb') as f:
              self.svm = pickle.load(f)

          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
            
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==2:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==3:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==4:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==5:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==6:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr =pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)
        elif catergory==7:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr =pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)


        elif catergory==8:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)


        elif catergory==9:
          #print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

        elif catergory==10:
          #print("--------",catergory,"-------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)


        # elif catergory==11:
        #   #print("--------",catergory,"---------")
        #   with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
        #       self.lr = pickle.load(f)
        #   with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
        #       self.svm = pickle.load(f)
        #   with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
        #       self.tfidf = pickle.load(f)
        #   with open(skill_list+str(catergory)+"_classes", 'rb') as f:
        #       self.classes_ = pickle.load(f)


        elif catergory==0:
          print("--------",catergory,"---------")
          with open(SKILL_PREDICT + str(catergory) + "_lr", 'rb') as f:
              self.lr = pickle.load(f)
          with open(SKILL_PREDICT + str(catergory) + "_svc", 'rb') as f:
              self.svm = pickle.load(f)
          #print(TF_IDF+str(catergory)+"_tfidf")
          with open(TF_IDF+str(catergory)+"_tfidf", 'rb') as f:
              self.tfidf = pickle.load(f)
          with open(skill_list+str(catergory)+"_classes", 'rb') as f:
              self.classes_ = pickle.load(f)

    def tf_df_vector(self, text):
        vectors = self.tfidf.transform([text])
        return vectors

    def clean_text(self, x):
        x = x.lower()
        x = re.sub('[^A-Za-z0-9]+', ' ', x)
        return x

    def text_normalization(self, text):
        # stopword removal and stemming
        string = ""
        for word in text.split():
            if not word in stop_words:
                # word=(sno.stem(word))
                string += word + " "
        return string

    def preprocessing(self, text):
        text = self.clean_text(text)
        self.text = self.text_normalization(text)
        return self.text

    def skill_predict_fun(self, text):
        text = self.preprocessing(text)
        vectors = self.tf_df_vector(text)
        # print(vectors.shape)
        # logistic regression
        skills_lr = self.lr.predict(vectors)
        # svm
        skills_svm = self.svm.predict(vectors)
        return skills_lr, skills_svm

    def brute_force_skill_matcher(self):
        skills_extract = []
        for skill_ in self.all_skills[self.listOfKeys]:
            blob = TextBlob(self.text)
            list_of_word = blob.ngrams(n=len(skill_.split(" ")))
            for n in list_of_word:
                cosine_scores = fuzz.ratio(" ".join(n).lower(), skill_.lower())
                if float(cosine_scores) >= 95:
                    skills_extract.append(skill_)
        return list(set(skills_extract))

    def class_mapping(self, predict_lr, predict_svm):
        predict_skills = []
        for x, y in zip(predict_lr[0], self.classes_):
            if x == 1:
                predict_skills.append(y)

        for x, y in zip(predict_svm[0], self.classes_):
            if x == 1:
                predict_skills.append(y)
        return list(set(predict_skills))


app = FastAPI()


@app.post("/predict")
async def predict(data: Text):
    data_dict = data.dict()
    text = data.content
    print(text)
    import pandas as pd
    '''
    text = """
    After Effects CC: The Complete Guide to After Effects CC	<p>Join this BRAND NEW Adobe After Effects CC course to have fun while learning quickly!</p><p>If you are looking to make your videos better, adding motion graphics and visual effects is the way to do that. After Effects CC is used by professionals across the world for every type of production from business & marketing videos, music videos to documentaries, feature films. This full course is the best way to jump right in and start making your own videos come to life.</p><p>What is this course all about?</p><p>Get ready to create motion graphics that will improve your video quality. If you've always wanted to learn how to use video effects and create custom motion graphics, you can learn After Effects CC right now.</p><p>Practice lessons will get you motivated and moving to your goals.</p><p>The lessons are meant to teach you how to think like a motion graphics/video effects artist. After Effects is a robust tool that is capable of creating almost any video effect out there. You will learn all of the basics, intermediate, and some advanced techniques, from working with shapes, text, and textures to video effects, transitions, and 3d camera.</p><p>Here is a glimpse of what we'll be covering:</p><p>Knowing After Effects CC is a great skill to have that is in demand and highly marketable. I have landed many jobs with the skills that I teach you in this course.</p>
    """
    
    csv_file = pd.read_csv(original_folder + File_name)
    csv_file = csv_file[['No', 'Name of the course', 'Description']]
    csv_file['text'] = csv_file['Name of the course'] + \
        ' ' + csv_file['Description']
    file_name = f"predicted_csv/Not_Predicted/{File_name}"
    file_name = file_name.replace(
        ".csv", "")
    csv_file[csv_file.isna()].to_csv(f"{file_name}_missing.csv", index=False)
    csv_file = csv_file[csv_file['text'].notna()]
    '''
    #print(text)
    skills_topics_ = pd.read_csv(MODEL_PATH + "topic-category-skills.csv")

    import ast

    skills_topics_['Skills'] = skills_topics_[
        'Skills'].apply(lambda x: x.split(","))
    cat_classifier = catergory_classifier()
    skill_obj = predict_skills_class()

    import time
    import math

    model_skills_eduonix = []
    skills_matcher = []
    combine_skills = []
    catergory_list = []

    # for text in tqdm(csv_file['text']):
    # print("========",text,"========")
    try:
        catergory = cat_classifier.predict(text)
        if not math.isnan(catergory):
            catergory_list.append(catergory)
            skill_obj.load_pickle(catergory)
            predict_lr, predict_svm = skill_obj.skill_predict_fun(text)
            predict_skills = skill_obj.class_mapping(predict_lr, predict_svm)
            model_skills_eduonix.append(predict_skills)
            skill_matcher = skill_obj.brute_force_skill_matcher()
            skills_matcher.append(skill_matcher)
            combine_skills.append(list(set(predict_skills + skill_matcher)))
        else:
            catergory_list.append(np.nan)
            model_skills_eduonix.append(np.nan)
            skills_matcher.append(np.nan)
            combine_skills.append(np.nan)
    except Exception as e:
        print(e)
        print("----<>", text)

    # print(catergory_list)
    # print(combine_skills)

    skill_rename = pd.read_csv(MODEL_PATH + "new_skill_rename.csv")
    skills_old = list(skill_rename["Old skills"].values)
    Edited_skills = list(skill_rename["Edited skills"].values)
    skill_rename_dict = {}
    for old_sk, edit_sk in zip(skills_old, Edited_skills):
        skill_rename_dict[old_sk] = edit_sk
    csv_skills_list = []

    row_csv_skill = []
    print(combine_skills)
    for sk in combine_skills[0]:
        if sk in skills_old:
            # print(sk,skill_rename_dict[sk])
            row_csv_skill.append(skill_rename_dict[sk])
        else:
            row_csv_skill.append(sk)
    # csv_skills_list.append(row_csv_skill)
   # print(row_csv_skill)

    import ast
    from tqdm import tqdm

    inv_mapping = {value: key for (key, value) in MAPPING.items()}

    def rations_(a_skills, b_skills):
        sum = 0
        for skill_a in a_skills:
            for skill_b in b_skills:
                if fuzz.ratio(skill_a, skill_b) >= 90:
                    sum += fuzz.ratio(skill_a, skill_b)
        return (sum)

    def predict_topics(skills_topics):
        topic_lists = []
        print("--------------------PREDICT TOPICS--------------------")
        # print(type(skill))
        catergory = int(catergory_list[0])
        lists = []

        if int(catergory) == 2:
            skills_topics = skills_topics_[
                skills_topics_['Category'] == inv_mapping[1]]
        else:
            skills_topics = skills_topics_[skills_topics_[
                'Category'] == inv_mapping[catergory]]

        topic_list = skills_topics['Topic'].values

        for skill_topic in skills_topics['Skills']:
            lists.append(rations_(row_csv_skill, skill_topic))

        first_max_value = max(lists)
        if first_max_value == 0:
            topic_lists.append("")
        else:
            predicted_index_first = lists.index(max(lists))
            first_topics_name = topic_list[predicted_index_first]
            lists[predicted_index_first] = 0
            second_max_value = max(lists)
            if second_max_value > 0:
                predicted_index_second = lists.index(max(lists))
                second_topics_name = topic_list[predicted_index_second]
                topic_lists.append([first_topics_name, second_topics_name])
            else:
                topic_lists.append([first_topics_name])
        return topic_lists
        # edu["Topics"] = topic_lists

    topic = predict_topics(skills_topics_)
    '''
    csv_file['Topics'].isnull().sum()
    csv_file['Skills'] = csv_file['Skills'].apply(lambda x: ",".join(x))
    csv_file['Topics'] = csv_file['Topics'].apply(lambda x: ",".join(x))
    csv_file['Topics'].replace('', np.nan)

    file_name = f"predicted_csv/{File_name}"
    file_name = file_name.replace(".csv", "")
    csv_file.to_csv(f"{file_name}_predicted.csv", index=False)
    '''
    if len(row_csv_skill) != 0 and len(topic[0]) != 0:
        return ",".join(row_csv_skill), ",".join(topic[0])
    elif len(row_csv_skill) == 0 and len(topic[0]) == 0:
        return "", ""
    elif len(row_csv_skill) != 0 and len(topic[0]) == 0:
        return ",".join(row_csv_skill), ""
    else:
        return "", ",".join(topic[0])
