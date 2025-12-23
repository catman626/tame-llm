import torch

# 创建一个测试张量 (4D)
a = torch.randn(2, 3, 3000, 4)

# 方法一：使用 None (隐式)
indices1 = (None, None, slice(0, 2048), None)
result1 = a[indices1]

# 方法二：使用 slice(None) (显式)
indices2 = (slice(None), slice(None), slice(0, 2048), slice(None))
result2 = a[indices2]

# 验证结果是否完全相同
print(f"形状相同: {result1.shape == result2.shape}") # 输出: True
print(f"数值相同: {torch.equal(result1, result2)}")  # 输出: True



print("hello world")


for r in range(10):
    print("r, ")

