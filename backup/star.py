import copy
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
    star_pos = set()
    stack = list()
    stack.append(star_pos)
    for _ in range(10000):
        if stack:
            sp = stack.pop()
            if len(sp) == 10:
                result.append(sp)
                continue
            for po in range(25):
                if collison(sp, po):
                    continue
                store = copy.copy(sp)
                store.add(po)
                store = sorted(store)
                if store not in stack + result:
                    stack.append(set(sorted(store)))
        else:
            break
    print(len(result))


def collison(line, point):
    line.add(point)
    exist_points = line
    flag = list(
        filter(lambda line: len(line.intersection(exist_points)) > 2, graph))
    if flag:
        return False


if __name__ == "__main__":
    main()
