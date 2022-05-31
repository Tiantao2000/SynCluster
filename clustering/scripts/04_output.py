
import pandas as pd
import pickle


#hide
def clu_new(cluster_type):
    k = 0
    newlist = {}
    for i in cluster_type:
        for po in i:
            newlist[po] = k
        k+=1
    test1=sorted(newlist.items(), key=lambda x: x[0])
    choose = [i[1] for i in test1]
    return choose

def cluster_the_small(cluster_dict):
    new_dict = {}
    for key,item in cluster_dict.items():
        big_list = []
        small_list = []
        for i,c in enumerate(item):
            sz = len(c)
            if sz < 50:
                newlist = [i for i in c]
                small_list = small_list + newlist
            else:
                big_list.append(c)
        big_list.append(tuple(small_list))
        new_dict[key]=tuple(big_list)
    return new_dict


def main(system = "uspto_50k"):
    data1=pd.read_pickle("data/"+system+"_fp.pkl")
    train_valid_data = pd.concat([data1[data1["split"] == "train"], data1[data1["split"] == "valid"]])
    test_data = data1[data1["split"] == "test"]

    #cut_off = 0.4
    #cut_off = 0.5
    #with  open(r"data/cluster_dict_" + str(cut_off) + ".pkl", 'rb')  as f1:
        #cluster_dict = pickle.load(f1)
    #cluster_dict = cluster_the_small(cluster_dict)
    #for i in cluster_dict.keys():
        #selected = clu_new(cluster_dict[i])
        #choose = i+"_cutoff_"+str(cut_off)
        #data1[choose] = selected

    # cut_off = 0.5
    cut_off = 0.6
    with  open(r"data/cluster_dict_" + str(cut_off) + ".pkl", 'rb')  as f1:
        cluster_dict = pickle.load(f1)
    cluster_dict = cluster_the_small(cluster_dict)
    for i in cluster_dict.keys():
        selected = clu_new(cluster_dict[i])
        choose = i+"_cutoff_"+str(cut_off)
        train_valid_data[choose] = selected

        # cut_off = 0.5
    #cut_off = 0.7
    #with  open(r"data/cluster_dict_" + str(cut_off) + ".pkl", 'rb')  as f1:
        #cluster_dict = pickle.load(f1)
    #cluster_dict = cluster_the_small(cluster_dict)
    #for i in cluster_dict.keys():
        #selected = clu_new(cluster_dict[i])
        #choose = i+"_cutoff_"+str(cut_off)
        #data1[choose] = selected
    #save_the_csv
    train_valid_data.to_csv("data/cluster_train_valid_50k.csv")
    test_data.to_csv("data/cluster_test_50k.csv")


if __name__ == '__main__':
    main()