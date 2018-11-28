from corel.model import *

def get_reader(fname):
	csvfile = open(fname)
	reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	next(reader, None)
	return reader

def train_test_compose(grad_type):
    f1 = train_model(get_reader('data/ccpp1.csv'), grad_type)
    #print('Trained 1, dataset 1')
    test1_1 = test_model(f1, get_reader('data/ccpp1.csv'))
    test1_2 = test_model(f1, get_reader('data/ccpp2.csv'))
    print('Trained with D1, Tested with D1: ' + str(test1_1) + ', D2: ' + str(test1_2) )

    f2 = train_model(get_reader('data/ccpp2.csv'), grad_type)
    test2_1 = test_model(f2, get_reader('data/ccpp1.csv'))
    test2_2 = test_model(f2, get_reader('data/ccpp2.csv'))
    print('Trained with D2, Tested with D1: ' + str(test2_1) + ', D2: ' + str(test2_2) )

    f = Model.compose(f1,f2)
    test = test_model(f, get_reader('data/ccpp.csv'))
    print('Composed, Tested with D: ' + str(test) )

def train_test(grad_type):
    f0 = train_model(get_reader('data/ccpp.csv'), grad_type)
    test0 = test_model(f0, get_reader('data/ccpp.csv'))
    print('Trained with D, tested with D: '+ str(test0))
    
    
# print('Train with SGD')
# train_test(GradType.SGD)
# print('Train with raw 2nd moment SGD')
# train_test(GradType.MOM)
# print('Train with momentum and variance')
# train_test(GradType.VAR)

# print('Compose with SGD')
# train_test_compose(GradType.SGD)
# print('Compose with raw 2nd moment SGD')
# train_test_compose(GradType.MOM)
print('Compose with momentum and variance')
train_test_compose(GradType.VAR)