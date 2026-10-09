import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import os, math, time, datetime
import scipy
import scipy.io
from scipy.io import wavfile
import numpy as np
import soundfile as s
from sinctdnn import SincTDNN


def ReadList(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig


def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, lab_dict_2, age_sex_to_lab, fact_amp):

    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    # rand_amp_arr = np.random.uniform(1.0-fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):

        # select a random sentence from the list  (joint distribution)
        [fs, signal] = scipy.io.wavfile.read(
            data_folder+wav_lst[snt_id_arr[i]])
        signal = signal.astype(float)/32768

        # accesing to a random chunk
        snt_len = signal.shape[0]
        # randint(0, snt_len-2*wlen-1)
        # snt_beg = np.random.randint(snt_len-wlen-1)
        snt_beg = 0
        snt_end = snt_beg+wlen

        if len(signal) < snt_end:
            signal = np.concatenate((signal, np.zeros(wlen - len(signal))))

        # sig_batch[i, :] = signal[snt_beg:snt_end]*rand_amp_arr[i]
        sig_batch[i, :] = signal[snt_beg:snt_end]

        age = lab_dict[wav_lst[snt_id_arr[i]]]
        sex = lab_dict_2[wav_lst[snt_id_arr[i]]]

        lab_batch[i] = age_sex_to_lab[age][sex]

    inp = torch.from_numpy(sig_batch).float(
    ).cuda("cuda").contiguous().unsqueeze(dim=2)  # Current Frame
    lab = torch.from_numpy(lab_batch).float().cuda("cuda").contiguous()

    return inp, lab


def main():
    model = SincTDNN()
    model_flag = 0
    restart = 0
    lr = 0.001
    lr_mag = 0.1

    if model_flag == 0:
        load_path = '/home/s246245/s226059/workspace/self_sinctdnn/save_model/embedding.pth'
        load_weights = torch.load(load_path)
        model.load_state_dict(load_weights)

        classes = 155
        model.add_module('output', nn.Linear(512, classes))

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)
        # model = torch.nn.DataParallel(model)
        # optimizer_model = optim.RMSprop(model.parameters(), lr=lr, alpha=0.95, eps=1e-8)
        optimizer_model = optim.Adam(model.parameters(), lr=lr)
        
    elif model_flag == 1:
        classes = 155
        model.add_module('output', nn.Linear(512, classes))
        load_path = '/home/s246245/s226059/workspace/self_sinctdnn/save_model/best20220903-073210/model_raw_47'
        load_saved_model = torch.load(load_path)
        load_weights = load_saved_model['model_state_dict']
        model.load_state_dict(load_weights)
        
        # optimizer_model = optim.RMSprop(model.parameters(), lr=lr, alpha=0.95, eps=1e-8)
        optimizer_model = optim.Adam(model.parameters(), lr=lr)
        optimizer_model.load_state_dict(load_saved_model['optimizer_state_dict'])
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        for state in optimizer_model.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        model.to(device)
        if restart == 1:
            epoch = load_saved_model['epoch']
            loss = load_saved_model['loss']
    else:
        classes = 155
        model.add_module('output', nn.Linear(512, classes))

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)
        optimizer_model = optim.RMSprop(model.parameters(), lr=lr, alpha=0.95, eps=1e-8)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_model, milestones=[50,100], gamma=lr_mag)
    cost = nn.CrossEntropyLoss()
    N_epochs = 500
    N_eval_epoch = 1
    
    batch_size = 64
    N_batches = int(151890/batch_size)
    data_folder = '/home/s246245/s226059/workspace/data/wav/'
    wav_lst_tr = ReadList('/home/s246245/s226059/workspace/self_sinctdnn/label/training_scp.txt')
    snt_tr = len(wav_lst_tr)
    N_snt = snt_tr
    fs = 16000
    duration = 4.0
    cw_len = 200
    cw_shift = 10


    lab_dict = np.load(
        '/home/s246245/s226059/workspace/self_sinctdnn/label/training_age_dict.npy', allow_pickle=True).item()
    print("lab_dict:\n",len(lab_dict))

    lab_dict_2 = np.load(
        '/home/s246245/s226059/workspace/self_sinctdnn/label/training_sex_dict.npy', allow_pickle=True).item()
    print("lab_dict_2:\n",len(lab_dict_2))
    age_sex_to_lab = np.load(
        '/home/s246245/s226059/workspace/self_sinctdnn/label/age_sex_to_lab.npy', allow_pickle=True).item()
    print("age_sex_to_lab\n:",len(age_sex_to_lab))
    rakuten_label = '/home/s246245/s226059/workspace/data/rakuten_age/16kHz/label.csv'


    # as_to_lab = '/home/s226059/workspace/self_sinctdnn/training_aslabel.txt'
    # data_folder_test = '/home/s226059/workspace/data/age_training/wav/test/'
    # wav_lst_te = ReadList('/home/s226059/workspace/self_sinctdnn/testing_scp.txt')
    # snt_te=len(wav_lst_te)


    seed = 1234
    output_folder = '/home/s246245/s226059/workspace/self_sinctdnn/save_model/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(output_folder)

    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder) 
    
    logging_folder = '/home/s246245/s226059/workspace/self_sinctdnn/log/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(logging_folder)

    try:
        os.stat(logging_folder)
    except:
        os.mkdir(logging_folder) 

    # setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Converting context and shift in samples
    # wlen=int(fs*cw_len/1000.00)
    wlen=int(fs*duration)
    wshift=int(fs*cw_shift/1000.00)
    

    def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

    start = time.time()

    with open(os.path.join(logging_folder, "train.log"), mode='a', newline='') as f:
            f.write("full epoch %i, lr=%f, (%s)" % (N_epochs, lr, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    print("Start training")
    for epoch in range(N_epochs):
        
        test_flag = 0
        model.train()

        loss_sum = 0
        err_sum = 0

        if restart == 1:
            loss = loss


        # for i in range(N_batches):

        #     [inp, lab] = create_batches_rnd(
        #         batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, lab_dict_2, age_sex_to_lab, 0.2)
        #     pout = (model(inp))

        #     pred = torch.max(pout, dim=1)[1]
        #     loss = cost(pout, lab.long())
        #     err = torch.mean((pred != lab.long()).float())

        #     optimizer_model.zero_grad()

        #     loss.backward()
        #     optimizer_model.step()
        #     scheduler.step()

        #     loss_sum = loss_sum+loss.detach()
        #     err_sum = err_sum+err.detach()

        # loss_tot = loss_sum/N_batches
        # err_tot = err_sum/N_batches
        # checkpoint = {'model_par': model.state_dict()}
        # # torch.save(checkpoint, output_folder+'/model_raw_2.pkl')
        # torch.save({'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer_model.state_dict(),
        #             'loss': loss,},
        #             output_folder+'/model_raw_'+str(epoch))

        # # print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
        # print("epoch %i, lr=%f, (%s), loss_tr=%f err_tr=%f" % (epoch, lr, timeSince(start), loss_tot, err_tot))
        # with open(os.path.join(logging_folder, "train.log"), mode='a', newline='') as f:
        #     f.write("\nepoch %i, lr=%f, (%s), loss_tr=%f err_tr=%f" % (epoch, lr, timeSince(start), loss_tot, err_tot))



    ######################## ここまでコメントアウトしてみた #############################

        # if epoch % N_eval_epoch == 0:

        #     model.eval()
        #     test_flag = 1
        #     loss_sum = 0
        #     err_sum = 0
        #     err_sum_snt = 0

        #     [inp, lab] = create_batches_rnd(
        #         1, data_folder_test, wav_lst_te, snt_te, wlen, lab_dict_test, age_to_lab, 0.2)
            
        #     model.eval()

        # else:
        #     print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))

        # if epoch % N_eval_epoch == 0:

        #     model.eval()
        #     test_flag = 1
        #     loss_sum = 0
        #     err_sum = 0
        #     err_sum_snt = 0

        #     with torch.no_grad():
        #         for i in range(snt_te):

        #             # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
        #             # signal=signal.astype(float)/32768

        #             [signal, fs] = sf.read(data_folder_test + wav_lst_te[i])

        #             signal = torch.from_numpy(signal).float().cuda().contiguous()
        #             lab_batch = age_to_lab[lab_dict_test[wav_lst_te[i]]]

        #             # split signals into chunks
        #             beg_samp = 0
        #             end_samp = wlen

        #             N_fr = int((signal.shape[0]-wlen)/(wshift))

        #             sig_arr = torch.zeros(
        #                 [Batch_dev, wlen]).float().cuda().contiguous()
        #             lab = Variable(
        #                 (torch.zeros(N_fr+1)+int(lab_batch)).cuda().contiguous().long())
        #             pout = Variable(torch.zeros(
        #                 N_fr+1, 83).float().cuda().contiguous())
        #             count_fr = 0
        #             count_fr_tot = 0
        #             while end_samp < signal.shape[0]:
        #                 sig_arr[count_fr, :] = signal[beg_samp:end_samp]
        #                 beg_samp = beg_samp+wshift
        #                 end_samp = beg_samp+wlen
        #                 count_fr = count_fr+1
        #                 count_fr_tot = count_fr_tot+1
        #                 if count_fr == Batch_dev:
        #                     inp = Variable(sig_arr).unsqueeze(dim=2)
        #                     pout[count_fr_tot-Batch_dev:count_fr_tot,
        #                         :] = model(inp)
        #                     count_fr = 0
        #                     sig_arr = torch.zeros(
        #                         [Batch_dev, wlen]).float().cuda().contiguous()

        #             if count_fr > 0:
        #                 inp = Variable(sig_arr[0:count_fr]).unsqueeze(dim=2)
        #                 pout[count_fr_tot-count_fr:count_fr_tot,
        #                     :] = model(inp)

        #             pred = torch.max(pout, dim=1)[1]
        #             loss = cost(pout, lab.long())
        #             err = torch.mean((pred != lab.long()).float())

        #             [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
        #             err_sum_snt = err_sum_snt+(best_class != lab[0]).float()

        #             loss_sum = loss_sum+loss.detach()
        #             err_sum = err_sum+err.detach()

        #         err_tot_dev_snt = err_sum_snt/snt_te
        #         loss_tot_dev = loss_sum/snt_te
        #         err_tot_dev = err_sum/snt_te

        #     print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
        #         epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        #     with open(output_folder+"/res.res", "a") as res_file:
        #         res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
        #             epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        #     checkpoint = {'model_par': model.state_dict()}
        #     torch.save(checkpoint, output_folder+'/model_raw.pkl')

        # else:
        #     print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))


if __name__ == '__main__':
    main()