import re

mySent = 'This book is the best book on Python on M.L. I have ever said eyes upon.'
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
print [tok.lower() for tok in listOfTokens if len(tok)>0]
