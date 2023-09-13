import json
import sys


def read_qindenton_json(filename):
   data = []

   with open(filename, "r") as f:

      jstring = ""
      for l in f:
         if l.startswith("#"):
            if l.endswith("End JSON\n"):
               jstring += "}"
               break
            else:
               jstring += l[1:]

      preprocessed = []
      for l in f:
         preprocessed.append(l.split())    # fill preprocessed[] with a list of lists (of values) for every line in a file

      j = json.loads(jstring)             # convert big file header string into a json object
      # realkeys = list(j.keys())[1:]       # top-level keys (e.g., "G" (not "G1" "G2" or "G3"))
      realkeys = ["G", "BzIMF", "ByIMF", "Pdyn", "DateTime", "Dst"]

      data = [{} for _ in preprocessed]

      for key in realkeys:
         # single column
         if not "DIMENSION" in j[key]:
            column = int(j[key]["START_COLUMN"])
            for i in range(len(data)):
               data[i][key] = preprocessed[i][column]
         # multi-dimensional
         else:
            column = int(j[key]["START_COLUMN"])
            elements = j[key]["ELEMENT_NAMES"]
            #print(elements)
            for i in range(len(data)):
               data[i][key] = {}
               for k, e in enumerate(elements):
                  try:
                     data[i][key][e] = preprocessed[i][column + k]
                  except IndexError:
                     print("ERROR: INDEX OUT OF RANGE")
                     print("i = ", i)
                     print("column = ", column)
                     print("k = ", k)
                     print("len(preprocessed[i]) = ", len(preprocessed[i]))
                     print("len(elements) = ", len(elements))
                     print("len(data) = ", len(data))
                     print("len(data[i]) = ", len(data[i]))
                     print("data[i] = ", data[i])
                     print("data[i][key] = ", data[i][key])
                     print("data[i][key][e] = ", data[i][key][e])
                     continue

   return data

if __name__ == "__main__":
   if len(sys.argv) != 2:
      print("FORGOT FILE NAME")
      exit(1)

   filename = sys.argv[1]

   #print(read_qindenton_json(filename))