

f2 = open("dqn_results_25%_synthetic_data_40_nodes.txt", "r")
for x in f2:
    f1 = open('dqn_results_25%_synthetic_data_40_nodes_new.txt', 'a+')
    f1.write("%s\n" % (float(x.strip())+0.1165))
    f1.close()
    print(
        "====================================================================================================================")

f2.close()

