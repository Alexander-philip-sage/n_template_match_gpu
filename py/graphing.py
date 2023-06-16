import matplotlib.pyplot as plt
import pandas as pd

scipy = pd.read_csv("tm_timing_scipy.csv")
assert len(scipy['accuracy'].unique())==1
scipy_N = pd.read_csv("tm_timing_N_scipy.csv")
opencv = pd.read_csv("tm_timing_opencv_cpu.csv")
assert len(opencv['accuracy'].unique())==1
opencv_N = pd.read_csv("tm_timing_N_opencv.csv")
tf_N = pd.read_csv("tm_timing_N_tf_batch.csv")
tf_batch= pd.read_csv("tm_timing_tf_batch.csv")

def scaling_search_window(df_list, label_list):
    template_size=800
    for df, label in zip(df_list,label_list):
        sub800 = df[df['template_size']==template_size]
        plt.plot(sub800['search_window_size'].to_numpy(), sub800['time'].to_numpy(), label=label)
    plt.xlabel("search window size")
    plt.ylabel("time per pair")
    plt.title("time per image template pair. after 10 pairs. template:"+str(template_size))
    plt.legend()
    fig = plt.gcf()
    fig.savefig("scaling_search_window.png")
    plt.close()
def scaling_template(df_list, label_list):
    search_window_size = 8000
    for df, label in zip(df_list,label_list):
        sub800 = df[df['search_window_size']==search_window_size]
        plt.plot(sub800['template_size'].to_numpy(), sub800['time'].to_numpy(), label=label)
    plt.xlabel("template_size")
    plt.ylabel("time per pair")
    plt.title("time per image template pair. after 10 pairs. search_window:"+str(search_window_size))
    plt.legend()
    fig = plt.gcf()
    fig.savefig("scaling_template.png")
    plt.close()

def scaling_pairs(df_list, label_list):
    for df, label in zip(df_list,label_list):
        plt.plot(df['N-pairs'].to_numpy(),df['time'].to_numpy() , label=label)
        search_window = df['search_window_size'].iloc[0]
        template= df['template_size'].iloc[0]
    plt.xlabel("N-pairs")
    plt.ylabel("time per pair")
    plt.title("time scaling with number of pairs. template {} search window {}".format(template, search_window))
    plt.legend()
    fig = plt.gcf()
    fig.savefig("scaling_n_pairs.png")
    plt.close()

scaling_search_window([scipy, opencv, tf_batch], ['scipy', 'opencv', 'tf'])
scaling_template([scipy, opencv, tf_batch], ['scipy', 'opencv','tf'])
scaling_pairs([scipy_N, opencv_N, tf_N], ['scipy', 'opencv','tf'])
#scaling_pairs([scipy_N, opencv_N, tf_N], ['scipy', 'opencv', 'tf'])
#print(scipy)