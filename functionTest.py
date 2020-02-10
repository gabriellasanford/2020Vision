def f(x):
    return x+1

def g(x):
    return x*x

def h(x):
    return 2*x

for fn in [f, g, h]:
    print fn(3)

print {fn:fn(3) for fn in [f, g, h]}

