import csv

with open('call_analysis.csv', 'r', encoding= 'utf-8') as f:
    reader = csv.reader(f, delimiter= '|')
    for row in reader:
        print('|'.join(row))