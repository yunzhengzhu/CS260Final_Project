import csv
import sys

def transform(task, k = 3):
    task_dir = './'+ task +'/'
    
    source = []
    with open(task_dir+'source_raw.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_id, row in enumerate(csv_reader):
            for i in range(k):
                source.append([int(row[0]), row[1]])
                
    target = []
    with open(task_dir+'target_raw.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for i in range(1, k+1):
                target.append([int(row[0]), row[i]])
      
    # writing source data into the file 
    with open(task_dir + 'source.csv', 'w', newline ='') as output_f:     
        write = csv.writer(output_f)
        write.writerows(source)
      
    # writing target data into the file 
    with open(task_dir + 'target.csv', 'w', newline ='') as output_f:     
        write = csv.writer(output_f)
        write.writerows(target)
        
        
if __name__ == "__main__":
    transform('train')
