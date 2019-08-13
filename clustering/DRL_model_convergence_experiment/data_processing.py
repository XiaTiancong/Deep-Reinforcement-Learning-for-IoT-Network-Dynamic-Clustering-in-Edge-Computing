

f2 = open("DRL_model_25%_synthetic_data_40_nodes.txt", "r")
for x in f2:
    f1 = open('DRL_model_25%_synthetic_data_40_nodes_new.txt', 'a+')
    f1.write("%s\n" % (float(x.strip())+0.05))
    f1.close()
    print(
        "====================================================================================================================")

f2.close()

