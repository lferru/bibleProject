import csv
import numpy as np
import string

verses = []
chapters = []
books = []
b = 1
c = 1
chapterString = ''
bookString = ''
translat = str.maketrans('', '', string.punctuation)

with open('./t_asv.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:

    curr = row['t'].translate(translat)
    
    verses.append(curr.lower())
    
    if ( row['c'] != c ):
      print(row['c'])
      c = row['c']
      chapters.append(chapterString.strip())
      chapterString = ''

    chapterString += curr.lower() + ' '

    if ( row['b'] != b ):
      
      b = row['b']
      books.append(bookString.strip())
      bookString = ''

    bookString += curr.lower() + ' '

chapters.append(chapterString.strip())
books.append(bookString.strip())
vArr = np.array(verses)
cArr = np.array(chapters)
bArr = np.array(books)
np.save('./verses.npy', vArr)
np.save('./chapters.npy', np.delete(cArr, 0))
np.save('./books.npy', np.delete(bArr, 0))
