import numpy as np

path = "data/sequences/test_1day_fixed.npz"

data = np.load(
    path,
    allow_pickle=False
)

print("=" * 60)
print("TEST DATASET")
print("=" * 60)

print("File:", path)

print("\nArrays:")

for key in data.files:

    arr = data[key]

    print(
        f"{key:20s}",
        "shape =", arr.shape,
        "dtype =", arr.dtype,
        "size =", arr.size
    )

print("\n" + "=" * 60)