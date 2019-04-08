# IPython log file

ip_info = get_ipython().getoutput(u'ifconfig eth0 | grep "inet "')
ip_info[0].strip()
foo = 'ipython*'
get_ipython().system(u'ls $foo')
get_ipython().magic(u'alias ll ls -l')
ll /usr
get_ipython().magic(u'alias test_alias (cd ch08; ls; cd ..)')
test_alias
get_ipython().magic(u'bookmark db /home/wesm/Dropbox/')
get_ipython().magic(u'cd db')
get_ipython().magic(u'bookmark -l')
get_ipython().magic(u'run ch03/ipython_bug.py')
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)
get_ipython().magic(u'run ch03/ipython_bug.py')
def f(x, y, z=1): 
    tmp = x + y
    return tmp / z
debug(f, 1, 2, z=3)
get_ipython().magic(u'run -d ch03/ipython_bug.py')
get_ipython().magic(u'run -d ch03/ipython_bug.py')
get_ipython().magic(u'run -d -b2 ch03/ipython_bug.py')
import time
start = time.time()
for i in range(iterations):
# some code to run here
elapsed_per = (time.time() - start) / iterations
import time
start = time.time()
for i in range(iterations):
    # some code to run here
    elapsed_per = (time.time() - start) / iterations
get_ipython().magic(u'matplotlib inline')
a = 5
a
import numpy as np
data = {i : np.random.randn() for i in range(7)}
data
from numpy.random import randn
data = {i : randn() for i in range(7)}
print data
an_apple = 27
an_example = 42
an<Tab>
b = [1, 2, 3]
b.<Tab> #按下Tab键
import datetime
datetime.<Tab> #按下Tab键
book_scripts/<Tab> #按下Tab键
path = 'book_scripts/<Tab> #按下Tab键
get_ipython().magic(u'pinfo b')
#执行显示结果
def add_numbers(a, b):
    """
    Add two numbers together

    Returns
    -------
    the_sum : type of arguments
    """
    return a + b
get_ipython().magic(u'pinfo add_numbers')
#执行显示结果
get_ipython().magic(u'psearch np.*load*')
#执行显示结果
def f(x, y, z):
    return (x + y) / z

a = 5
b = 6
c = 7.5

result = f(a, b, c)
get_ipython().magic(u'run ipython_script_test.py')
c
result
x = 5
y = 7
if x > 5:
    x += 1
y = 8
get_ipython().magic(u'paste')
#终端可以
get_ipython().magic(u'cpaste')
#终端可以
get_ipython().magic(u'run ch03/ipython_bug.py')
a = np.random.randn(100, 100)
get_ipython().magic(u'timeit np.dot(a, a)')
get_ipython().magic(u'pinfo %reset')
#执行显示结果
ipython qtconsole --pylab=inline
#需安装PyQt或PySide
_i28
_28
exec _i28
get_ipython().magic(u'logstart')
ip_info = get_ipython().getoutput(u'ifconfig eth0 | grep "inet "')
ip_info[0].strip()
foo = 'ipython*'
get_ipython().system(u'ls $foo')
get_ipython().magic(u'alias ll ls -l')
ll /usr
get_ipython().magic(u'alias test_alias (cd ch08; ls; cd ..)')
test_alias
get_ipython().magic(u'bookmark db /home/wesm/Dropbox/')
2 ** 27
_
foo = 'bar'
foo
exec _i53
_i53
_53
exec _i53
get_ipython().magic(u'logstart')
ip_info = get_ipython().getoutput(u'ifconfig eth0 | grep "inet "')
ip_info[0].strip()
foo = 'ipython*'
get_ipython().system(u'ls $foo')
get_ipython().magic(u'logstart')
get_ipython().magic(u'logstart')
ip_info = get_ipython().getoutput(u'ifconfig eth0 | grep "inet "')
ip_info[0].strip()
foo = 'ipython*'
get_ipython().system(u'ls $foo')
get_ipython().magic(u'alias ll ls -l')
ll /usr
get_ipython().magic(u'alias test_alias (cd ch08; ls; cd ..)')
test_alias
get_ipython().magic(u'bookmark db /home/wesm/Dropbox/')
get_ipython().magic(u'cd db')
get_ipython().magic(u'bookmark -l')
get_ipython().magic(u'run ch03/ipython_bug.py')
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args, **kwargs)
run ch03/ipython_bug.py
#这部分是调试介绍
def f(x, y, z=1): 
    tmp = x + y
    return tmp / z
debug(f, 1, 2, z=3)
get_ipython().magic(u'run -d ch03/ipython_bug.py')
get_ipython().magic(u'run -d -b2 ch03/ipython_bug.py')
import time
start = time.time()
for i in range(iterations):
    # some code to run here
    elapsed_per = (time.time() - start) / iterations
import time
start = time.time()
iterations = 10
for i in range(iterations):
    # some code to run here
    elapsed_per = (time.time() - start) / iterations
strings = ['foo', 'foobar', 'baz', 'qux','python', 'Guido Van Rossum'] * 100000 
method1 = [x for x in strings if x.startswith('foo')] 
method2 = [x for x in strings if x[:3] == 'foo']
get_ipython().magic(u"time method1 = [x for x in strings if x.startswith('foo')]")
get_ipython().magic(u"time method2 = [x for x in strings if x[:3] == 'foo']")
get_ipython().magic(u"timeit [x for x in strings if x.startswith('foo')]")
get_ipython().magic(u"timeit [x for x in strings if x[:3] == 'foo']")
x = 'foobar'
y = 'foo'
get_ipython().magic(u'timeit x.startswith(y)')
get_ipython().magic(u'timeit x[:3] == y')
import numpy as np
from numpy.linalg import eigvals
def run_experiment(niter=100): K = 100
results = []
for _ in xrange(niter):
mat = np.random.randn(K, K)
max_eigenvalue = np.abs(eigvals(mat)).max() results.append(max_eigenvalue)
return results
some_results = run_experiment()
print 'Largest one we saw: %s' % np.max(some_results)
import numpy as np
from numpy.linalg import eigvals
def run_experiment(niter=100):
    K = 100
    results = []
    for _ in xrange(niter):
        mat = np.random.randn(K, K)
        max_eigenvalue = np.abs(eigvals(mat)).max()
        results.append(max_eigenvalue)
    return results
some_results = run_experiment()
print 'Largest one we saw: %s' % np.max(some_results)
python -m cProfile cprof_example.py
get_ipython().magic(u'prun -l 7 -s cumulative run_experiment()')
from numpy.random import randn
def add_and_sum(x, y):
    added = x + y
    summed = added.sum(axis=1) 
    return summed
def call_function():
    x = randn(1000, 1000)
    y = randn(1000, 1000) 
    return add_and_sum(x, y)
get_ipython().magic(u'run prof_mod')
c.TerminalIPythonApp.extensions = ['line_profiler']
x = randn(3000, 3000)
y = randn(3000, 3000)
get_ipython().magic(u'prun add_and_sum(x, y)')
get_ipython().magic(u'lprun -f func1 -f func2 statement_to_profile')
c.TerminalIPythonApp.extensions = ['line_profiler']
#c.TerminalIPythonApp.extensions = ['line_profiler']
#%run prof_mod
#这里我电脑也不行，搜不到这个文件
#%lprun -f func1 -f func2 statement_to_profile
#%lprun -f func1 -f func2 statement_to_profile
get_ipython().magic(u'lprun -f add_and_sum add_and_sum(x, y)')
#%lprun -f func1 -f func2 statement_to_profile
#%lprun -f add_and_sum add_and_sum(x, y)
#这里要联系上面的line_profile这个IPython扩展，书上没说
#%lprun -f add_and_sum -f call_function call_function()
#ipython notebook --pylab=inline
## 利用IPython提高代码开发效率的几点提示
import some_lib
x= 5
y = [1, 2, 3, 4]
result = some_lib.get_answer(x, y)
from my_functions import g
def f(x, y):
    return g(x + y)
def main(): 
    x= 6
    y = 7.5
    result = x + y
if __name__ == '__main__': 
    main()
class Message:
    def __init__(self, msg):
        self.msg = msg
x = Message('I have a secret')
x = Message('I have a secret')
x
class Message:
    def __init__(self, msg):
        self.msg = msg
    def __repr__(self):
        return 'Message: %s' % self.msg
x = Message('I have a secret')
x
