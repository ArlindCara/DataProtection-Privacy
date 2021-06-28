
## k-degree anonymity
### Install requirements
Per poter eseguire il file k-degree.py è necessario prima installare networkx e numpy
```
pip install -r requirements.txt
```

### Usage
Far partire da terminale il comando costituito da
1) python
2) nome file python da eseguire
3) valore di k degree
4) file csv da anonimizzare
```
usage: python k-degree.py k_value graph_to_anonymize.csv
```
#### Example
```
usage: python k-degree.py 3 Dataset/graph_friend_100_10_100.csv
```

#### Dataset
Nella cartella Dataset si trova un file csv per poter eseguire il programma:

- Dataset\graph_friend_100_10_100.csv
