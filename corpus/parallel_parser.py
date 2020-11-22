from bs4 import BeautifulSoup as bs
import csv
import os
import re


def parse_bible(path):
    content = []
    with open(path, "r") as file:
        content = file.read()
        bs_content = bs(content, "lxml")
        return bs_content


def get_verses(bible):
    return bible.find_all("seg", {"type": "verse"})


def get_ids_in_common(b1, b2):
    b1_ids = set([verse.get("id") for verse in b1])
    b2_ids = set([verse.get("id") for verse in b2])
    return b1_ids.intersection(b2_ids)


def get_verse(bible, id):
    return re.sub('[\\n\\t]', '', bible.find(id=id).text)


def write_parallel_bible(bible1, bible2, writepath):
    verses_b1 = get_verses(bible1)
    verses_b2 = get_verses(bible2)
    verse_ids = get_ids_in_common(verses_b1, verses_b2)

    print("Working")
    with open(writepath, "+w", newline='') as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["Nahuatl", "Spanish", "Verse"])
        for v_id in verse_ids:
            curr_row = []
            curr_row.append(get_verse(bible1, v_id))
            curr_row.append(get_verse(bible2, v_id))
            curr_row.append(v_id)
            # print(curr_row)
            writer.writerow(curr_row)
    print("done")


if __name__ == "__main__":
    bible1 = parse_bible("bible-corpus/bibles/Nahuatl-NT.xml")
    bible2 = parse_bible("bible-corpus/bibles/Spanish.xml")
    writepath = os.path.join(os.getcwd(), "parallel_nh-es.csv")

    write_parallel_bible(bible1, bible2, writepath)
