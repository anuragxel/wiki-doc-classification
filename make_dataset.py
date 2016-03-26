import csv
import bleach
import re

def remove_upper_bullshit(html):
    idx = html.find("<p>")
    return html[idx:]

def remove_lower_bullshit(html):
    idx = html.find('[edit] References')
    if idx:
        html = html[:idx]
    idx = html.find('[edit] External links')
    if idx:
        html = html[:idx]
    return html

def remove_whitespace(string):
    return re.sub('\s+', ' ', string).strip()

def remove_edit_and_citation_tag(string):
    string = re.sub('\[[0-9]*\]', '', string).strip()
    string = re.sub('\[citation needed\]', '', string).strip()
    return re.sub('\[edit\]', '', string).strip()

def remove_comment_tag(string):
    return re.sub('//', '', string).strip()

def replace_quotes(string):
    return re.sub('"', "'", string).strip()

def bleach_html(html):
    allowed_tags = []
    allowed_attributes = {}
    return bleach.clean(html, tags = allowed_tags, attributes = allowed_attributes, strip = True)

if __name__ == "__main__":
    root_path = 'documents/'
    with open('cleaned_tag.csv', 'rb') as csvfile:
        # (idx, title, identifier, chosen_tag) comprises a row
        spamreader = csv.reader(csvfile, delimiter = ',')
        for row in spamreader:
            index, title, identifier, tag = row
            html = open(root_path + identifier, 'r').readlines()
            html = "".join(line for line in html)
            html = remove_upper_bullshit(html)
            html = bleach_html(html)
            html = remove_lower_bullshit(html)
            html = remove_whitespace(html)
            html = remove_edit_and_citation_tag(html)
            html = replace_quotes(html) 
            html = remove_comment_tag(html)
            html = html.lower().encode('ascii', 'ignore')
            print '%s,"%s",%s,"%s",%s' % (index, title, identifier, html, tag)