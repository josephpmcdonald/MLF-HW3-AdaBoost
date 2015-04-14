from subprocess import call
import re

def formatData():
    call("rm lstrain.txt", shell=True)
    call("rm lstest.txt", shell=True)
    call("rm train.txt", shell=True)
    call("rm test.txt", shell=True)
    call("../libsvm-3.17/svm-scale -s params splice_noise_train.txt > lstrain.txt", shell=True)
    call("../libsvm-3.17/svm-scale -r params splice_noise_test.txt > lstest.txt", shell=True)

    with open('lstrain.txt','r') as fr:
    #with open('formatTest.txt','r') as fr:
        with open('train.txt','w') as fw:
            for line in fr:
                newline = line.translate(None, '+')
                List = re.split(' |:', newline)
                fw.write(List.pop(0)+' ')

                for ind in range(0,len(List),2):
                    List[ind]=''
                
                fw.write(' '.join(List)+'\n')

    with open('lstest.txt','r') as fr:
        with open('test.txt','w') as fw:
            for line in fr:
                newline = line.translate(None, '+')
                List = re.split(' |:', newline)
                fw.write(List.pop(0)+' ')

                for ind in range(0,len(List),2):
                    List[ind]=''
                
                fw.write(' '.join(List)+'\n')


if __name__ == "__main__":
    formatData()
