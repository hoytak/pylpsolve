# This is just a handy little script that prints out all the possible
# combinations of test cases.

from itertools import product

V = ["dl",
     "tnNla",
     "tnNl",
     "smalLA",
     "LA",
     "lLBA"]

title = "2dMatrixBlocks"

for i, t in enumerate(product(*V)):
    if t[0] == "d" and (t[1] in "la" or t[2] in "l"):
	continue

    if t[1] == "n" and t[3] == "s":
	continue
     
    if i % (len(V[-1])) == 0:
	print ""

    s = "".join(t)
    print "    def test%s_%s(self): self.check%s(\"%s\")" % (title,s,title,s)


