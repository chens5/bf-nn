import yaml

def main():
    dict = {}
    act = 'ReLU'
    bias = True
    depth = 1
    widths = [4, 8, 12, 16, 20, 24]
    widths = [8]
    epsilon = [1.0, 0.1, 0.01, 0.001, 0.0001]
    for i in range(len(widths)):
        for eps in epsilon:
            width = widths[i]
            dict[f'w-{width}-d-1-eps-{eps}'] = {'act': act, 
                                    'bias': bias, 
                                    'depth':depth, 
                                    'width': width,
                                    'epsilon_param': eps}
    file = open("../model-configs/configs-l0.yaml", 'w')
    result = yaml.dump(dict, file)
    file.close()
    return

if __name__ == '__main__':
    main()