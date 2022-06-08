import scm

model_save_dir = 'models/'
metrics_save_dir = 'results/'
scms_save_dir = 'scms/'

def get_train_epochs(dataset, model, trainer):
    if trainer[:4] == 'ALLR':
        epochs = {'bail': {'lin': 20, 'mlp': 500},
                 'compas': {'lin': 10, 'mlp': 20},
                 'german': {'lin': 40, 'mlp': 20},
                 'adult': {'lin': 20, 'mlp': 80},
                 'loan': {'lin': 20, 'mlp': 30}}
    elif trainer == 'ROSS':
        epochs = {'bail': {'lin': 40, 'mlp': 100},
                   'compas': {'lin': 20, 'mlp': 10},
                   'german': {'lin': 20, 'mlp': 20},
                   'adult': {'lin': 20, 'mlp': 80},
                   'loan': {'lin': 30, 'mlp': 20}}
    else:
        epochs = {'bail': {'lin': 200, 'mlp': 50},
                   'compas': {'lin': 100, 'mlp': 10},
                   'german': {'lin': 500, 'mlp': 20},
                   'adult': {'lin': 30, 'mlp': 30},
                   'loan': {'lin': 20, 'mlp': 100}}
    return epochs[dataset][model]

def get_lambdas(dataset, model_type, trainer):
    if trainer[:4] == 'ALLR':
        if model_type == 'lin':
            return {'compas': 0.1, 'german': 0.1, 'adult': 0.1, 'loan': 0.1, 'bail': 0.1}[dataset]
        elif model_type == 'mlp':
            return {'compas': 0.1, 'german': 0.5, 'adult': 0.5, 'loan': 0.01, 'bail': 0.01}[dataset]
    elif trainer == 'ROSS':
        return 0.8
    else:
        return 0

def get_recourse_hyperparams(trainer):
    # if trainer in ['ROSS', 'ALLR']:
    #     return {'lr': 0.1, 'lambd_init': 10.0, 'decay_rate': 0.9, 'outer_iters': 200, 'inner_iters': 50,
    #             'recourse_lr': 0.1}
    # else:
    return {'lr': 0.1, 'lambd_init': 1.0, 'decay_rate': 0.9, 'outer_iters': 100, 'inner_iters': 50, 'recourse_lr': 0.1}

def get_model_save_dir(dataset, trainer, model, random_seed, lambd=None, epochs=None):
    if trainer in ['ERM', 'AF']:
        model_dir = model_save_dir+'%s_%s_%s_s%d' % (dataset, trainer, model, random_seed)
    else:
        model_dir = model_save_dir+'%s_%s_%s_l%.3f_s%d' % (dataset, trainer, model, lambd, random_seed)

    if epochs is not None:
       model_dir += '_e' + str(epochs) + '.pth'
    return model_dir

def get_metrics_save_dir(dataset, trainer, lambd, model, epsilon, seed):
    if trainer in ['ERM', 'AF']:
        return metrics_save_dir + '%s_%s_%s_e%.3f_s%d' % (dataset, trainer, model, epsilon, seed)
    else:
        return metrics_save_dir + '%s_%s-%.3f_%s_e%.3f_s%d' % (dataset, trainer, lambd, model, epsilon, seed)

def get_tensorboard_name(dataset, trainer, lambd, model, train_epochs, learning_rate, random_seed):
    if trainer in ['ERM', 'AF']:
        return '%s_%s_%s_epochs%d_lr%.4f_s%d' % (dataset, trainer, model, train_epochs, learning_rate, random_seed)
    else:
        return '%s_%s-%.2f_%s_epochs%d_lr%.4f_s%d' % (dataset, trainer, lambd, model, train_epochs, learning_rate, random_seed)

def get_scm(model_type, dataset):
    if model_type == 'mlp' and dataset == 'loan':
        return scm.SCM_Loan()
    scms = {'adult': scm.Learned_Adult_SCM, 'compas': scm.Learned_COMPAS_SCM}
    if dataset in scms.keys():
        scmm = scms[dataset](linear=model_type=='lin')
        scmm.load(scms_save_dir+dataset)
        return scmm
    return None
