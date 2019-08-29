def split(m):
    if m:
        for k in range(m):
            split(k)
            split(m - k)
    else:
        return 0


split(10)
