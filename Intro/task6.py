def check(x: str, file: str):
    f = open(file, 'w')
    print(file)
    words = x.lower().split()

    res = {}

    for word in words:
        counter = 0
        for cur in words:
            if word == cur:
                counter += 1
        res[word] = counter

    res = dict(sorted(res.items()))
    for el in res:
        f.write(f'{el} {res[el]}\n')

    f.close()
