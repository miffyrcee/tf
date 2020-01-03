import string

a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y = range(
    25)
graph = [{a, c, e, f, g}, {a, b, x, m, l}, {a, v, w, t, r}, {d, c, b, v, u},
         {d, e, x, p, o}, {d, f, h, i, j}, {g, h, x, s, r}, {g, i, k, y, l},
         {j, k, x, w, u}, {j, y, m, n, o}, {l, n, p, q, r}, {o, q, s, t, u}]


def check_point(po, star_pos, stack):
    if po in star_pos:
        return False
    for st in stack:
        if set(st).difference(set(star_pos + [po])):
            continue
        else:
            return False
    for gh in graph:
        if len(gh.intersection(set(star_pos + [po]))) > 2:
            return False
    return True


def main():
    result = list()
    # star_pos = [s, h]
    star_pos = []
    stack = list()
    stack.append(star_pos)
    for _ in range(100000000):
        if stack:
            sp = stack.pop()  #取出未完成的star_pos
            if len(sp) == 10:
                print(sp)
                result.append(sp)
                continue
            for po in range(25):
                if check_point(po, sp, stack):  #检查点是否符合条件
                    stack.append(sp + [po])  #更新stack
        else:
            break
    print(len(result))  #


if __name__ == "__main__":
    main()
