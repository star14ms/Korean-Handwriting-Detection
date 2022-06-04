import numpy as np


# evaluation을 위해 필요한 함수 정의

def editDistance(r, h):

    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

# wer(wor error rate) 참조 링크
# https://docs.microsoft.com/ko-kr/azure/cognitive-services/speech-service/how-to-custom-speech-evaluate-data
def wer(r, h):

    # build the matrix
    d = editDistance(r, h)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result

def evaluation_metrics(pred_list, validation_dataset):
    return evaluate(pred_list, validation_dataset)


def evaluate(pred_list, validation_dataset):
    
    total_wer = 0
    
    for i in range(len(pred_list)) :
        wer_val = wer(validation_dataset.labels[i].split(), pred_list[i].split())
        total_wer += wer_val
    ret = total_wer / len(pred_list)
    return ret