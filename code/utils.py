# Utility functions
from imports_lib import *

def argAssigner(args):
    # TODO check the data types
    global lr,batch_size,init_Code,num_epochs,model_initializer,save_dir, verbose,debug,restart
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)
    save_dir = args.save_dir
    init_Code = int(args.init)
    # TODO check normal or uniform
    model_initializer = initializers.glorot_normal(seed=None) if init_Code == 1 else initializers.he_normal(seed=None)
    verbose = args.verbose
    restart = args.restart
    debug = args.debug

def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.0001, help='Initial learning rate (eta)', type=float)
    parser.add_argument('--batch_size',default=2, help='Batch size, -1 for vanilla gradient descent')
    parser.add_argument('--num_epochs',default=100, help='Saves model parameters in this directory')
    parser.add_argument('--init',default=1, help='Initializer: 1 for Xavier init and 2 for He init')
    parser.add_argument('--save_dir',default='./save_dir/', help='Saves model parameters in this directory')
    # Custom debugging args
    parser.add_argument('-d','--debug', help='For devs only, takes in no arguments', action="store_true")
    parser.add_argument('-v',"--verbose", help="Increase output verbosity",action="store_true")
    parser.add_argument('-r',"--restart", help="Restarts the network",action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print("verbosity turned on")
        for k in args.__dict__:
            print ('\x1b[6;30;42m' + str(k) + '\x1b[0m' + '\t\t' + str(args.__dict__[k]))

    return args, args.__dict__

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def ensure_dir(files):
    for f in files:
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    return 1
