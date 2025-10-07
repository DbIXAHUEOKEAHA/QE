nx, ny, nz = 8, 8, 1
weight = 1 / (nx * ny * nz)

# Open a file to write
with open("kpoints_grid.txt", "w") as f:
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                line = f"    {i/nx:.16f} {j/ny:.16f} {k/nz:.16f} {weight:.16f}\n"
                f.write(line)

print("K-points grid saved to kpoints_grid.txt")