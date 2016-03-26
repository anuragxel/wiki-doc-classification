import os
import sys
import xml.etree.ElementTree as ET
import operator

def get_categories(root):
    category = {}
    idx = 1
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            for tag in child[3][0]:
                # ---------------------------------------------------
                # HEREIN LIES THE MAGIC
                # ---------------------------------------------------
                if tag.text in category and not tag.text.isdigit():
                    category[tag.text] += 1
                else:
                    category[tag.text] = 1
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
            for tag in child[3][0]:
                # Most relevent tag present in our category set is chosen
                if tag.text in category:
                    chosen_tag = tag.text
            # If there is no chosen tag, it means that the document needs to be discarded
            if not chosen_tag:
                continue
            print '%d,"%s",%s,%s' % (idx, title, identifier, chosen_tag)
            idx += 1
        except:
            pass

if __name__ == "__main__":
    tag_tree = ET.parse('tag-data.xml')
    root = tag_tree.getroot()
    category = get_categories(root)
    sorted_category = sorted(category.items(), key=operator.itemgetter(1),reverse=True)
    #print len(sorted_category)
    category_set =  dict(sorted_category[:400])
    make_category_for_doc(root, category_set)