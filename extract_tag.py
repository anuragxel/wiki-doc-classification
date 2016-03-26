import os
import sys
import xml.etree.ElementTree as ET
import operator
import csv
import pprint
from nltk.stem.snowball import SnowballStemmer
import json

stupid_tags = [ '-', '*' ]

def get_categories(root):
    category = {}
    idx = 1
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            for tag_par in child[3]:
                tag = tag_par[0].text
                count = tag_par[1].text
                # Both the tag and count may be switched around,
                # swap to have correctness
                if not count.isdigit() and tag.isdigit():
                    tag,count = count,tag 
                if tag in category and not tag.isdigit():
                    category[tag] += 1
                else:
                    category[tag] = 1
            idx += 1
        except:
            pass
    return category

def make_category_for_doc(root, category):
    idx = 1
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            chosen_tag = None
            # tags are in sorted order
            # hence we iterate from top tag to less relevent tag automatically
            for tag_par in child[3]:
                tag = tag_par[0].text
                count = tag_par[1].text
                # Both the tag and count may be switched around,
                # swap to have correctness
                if not count.isdigit() and tag.isdigit():
                    tag,count = count,tag 
                # Most relevent tag present in our category set is chosen
                if tag in category and tag not in stupid_tags:
                    chosen_tag = tag
            # If there is no chosen tag, it means that the document needs to be discarded
            if not chosen_tag or chosen_tag in stupid_tags:
                continue
            print '%d,"%s",%s,%s' % (idx, title, identifier, chosen_tag)
            idx += 1
        except:
            pass

def extract_tag():
    tag_tree = ET.parse('tag-data.xml')
    root = tag_tree.getroot()
    category = get_categories(root)
    sorted_category = sorted(category.items(), key=operator.itemgetter(1),reverse=True)
    category_set =  dict(sorted_category[:472])
    make_category_for_doc(root, category_set)

def check_extracted_tag_csv(filename):
    categories = {}
    idx = 1
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            idx += 1
            if row[3] in categories:
                categories[row[3]] += 1
            else:
                categories[row[3]] = 1
    return (idx, categories)

def clean_up_tags():
    new_categories = {}
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    with open('top_tag.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            old_key = row[3]
            if old_key not in new_categories:
                new_categories[old_key] = stemmer.stem(old_key)
    
    with open('top_tag.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            print '%s,"%s",%s,%s' % (row[0],row[1],row[2], new_categories[row[3]])

    json.dump(new_categories, open('category_stem_dict', 'w'))

    

if __name__ == "__main__":
    ### 
    ### Use this to extract the tags from xml file. Then redirect to top_tag.csv
    ###
    #extract_tag()
    
    ###
    ### Check whether everything is fine. The tags are similar, so we need to clean up
    ###
    #c = check_extracted_tag_csv('top_tag.csv')
    #pprint.pprint(c[1])
    #print len(c[1])
    
    ###
    ### clean up tags by stemming the tags from the top_tags.csv and then redirect
    ###
    #clean_up_tags()

    ###
    ### Check whether everything is fine. The tags are similar, so we need to clean up
    ###
    c = check_extracted_tag_csv('cleaned_tag.csv')
    pprint.pprint(c[1])
    print len(c[1])

