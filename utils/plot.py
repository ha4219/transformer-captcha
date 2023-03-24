import matplotlib.pyplot as plt


def train_test_loss_acc_plot(train_loss, train_acc, test_acc):
    assert len(train_loss) == len(train_acc) == len(test_acc)

    length = len(train_acc)
    x = range(length)
    fig, ax1 = plt.subplots()
    ax1.plot(x, train_loss)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')

    ax2 = ax1.twinx()
    ax2.plot(x, train_acc)
    ax2.plot(x, test_acc)
    ax2.set_ylabel('acc')
    
    import time
    timestr = time.strftime("%Y%m%d%H%M%S")
    plt.savefig(f'vis/{timestr}.png')

    return


if __name__ == '__main__':
    train_test_loss_acc_plot(
        [1, 2, 1, 2],
        [0.9, 0.3, 0.2, 0.1],
        [0.1, 0.1, 0.2, 0.2],
    )