import jax_triton as jt
import inspect

print("triton_call signature:")
try:
    print(inspect.signature(jt.triton_call))
except Exception as e:
    print(e)

print("\nDir jt:")
print(dir(jt))
