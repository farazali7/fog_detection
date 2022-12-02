from config import cfg
from main import train_loso
import logging
import optuna as opt
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import joblib

def objective(trial):
    
    #number of modalities (5 different modalities)
    modalities_list = []

    EMG = trial.suggest_int('EMG', 0, 1)
    # if EMG == 1:
    #     modalities_list.append("EMG")
    modalities_list.append("EMG")
    
    ECG = trial.suggest_int('ECG', 0, 1)
    # if ECG == 1:
    #    modalities_list.append("ECG")
    modalities_list.append("ECG")

    ACC = trial.suggest_int('ACC', 0, 1)
    if ACC == 1:
        modalities_list.append("ACC")

    GYR = trial.suggest_int('GYR', 0, 1)
    if GYR == 1:
        modalities_list.append("GYR")

    NC_SC = trial.suggest_int('NC/SC', 0, 1)
    if NC_SC == 1:
        modalities_list.append("NC/SC")
    
    cfg['MODALITIES'] = modalities_list

    #number of windows (1 - 5)
    windows_num = trial.suggest_int('num_windows', 1, 5, step=1)
    cfg['N_WINDOWS']  = windows_num

    #win.len (3.2, 6.4, 12.8)  - number of samples would be a power to 2
    window_len = trial.suggest_categorical('window_length', [3.2, 6.4, 12.8])
    cfg['WIN_LENGTH'] = window_len

    #percentage overlap(0, 0.25, 0.50, 0.75)
    percent_overlap = trial.suggest_float('percentage overlap', 0, 0.75, step=0.25)
    cfg['OVERLAP'] =  percent_overlap

    #number of heads **has to be even(2, 4, 8, 16)
    heads_num_exp = trial.suggest_int('number of heads', 1, 4, step=1)
    heads_num = 2**heads_num_exp
    cfg['N_HEADS'] = heads_num

    #number of encoder layers (1-12)
    encoder_layer_num = trial.suggest_int('number of encoder layers', 1, 12, step=1)
    cfg['N_ENC_LAYERS'] = encoder_layer_num

    #number of filters
    num_filters = trial.suggest_int("num_channels", 64, 512, log=True)
    cfg['N_MAX_CONV_FILTERS'] = num_filters

    #sample rate (Start with 40, but let try higher as well)
    cfg['SAMPLE_RATE'] = 40
    

    #Start with constant values for batch size, learning rate, betas, epsilon, weight decay
    
    # print(cfg)
    train_mean_acc, val_mean_acc = train_loso(subjects=cfg['SUBJECTS'], modalities=cfg['MODALITIES'], sample_rate=cfg['SAMPLE_RATE'], win_len=cfg['WIN_LENGTH'], overlap=cfg['OVERLAP'], n_windows=cfg['N_WINDOWS'], num_head=cfg['N_HEADS'], num_ec_layers=cfg['N_ENC_LAYERS'], num_filters=cfg['N_MAX_CONV_FILTERS'])
    run["train/accuracy"].log(train_mean_acc)
    run["validation/accuracy"].log(val_mean_acc)
    print("logged")
    # if trial.should_prune():
    #         raise opt.TrialPruned()
    return val_mean_acc

# #Load previous study?
# experiment_one = joblib.load("study.pkl")

# #Add stream handler to show of trails were prunned. 
# opt.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

run = neptune.init(
    project="SYDE599",
	api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZWUyZjYwYi1lZGE3LTQ2NGUtYTliOC03OWEzZTRlYjRiMDYifQ=="
)

neptune_callback = optuna_utils.NeptuneCallback(run)


study = opt.create_study(study_name='SYDE599_Run', direction='maximize')
study.optimize(objective, n_trials=20, callbacks=[neptune_callback])
print(" Value: ", study.best_trial.value)
print(" Params: ")

joblib.dump(study, "study.pkl")

optuna_utils.log_study_metadata(
    study,
    run
)

# To resume study
# study = joblib.load("study.pkl")


# experiment_one = opt.create_study(direction='maximize', pruner=opt.pruners.MedianPruner())


# joblib.dump(experiment_one, "study.pkl")