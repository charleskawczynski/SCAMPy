import os

def get_all_files_in_path_recursive(root_path, names_to_exclude, ext_include):
    f = [[subdir + os.sep + x for x in files] for subdir, dirs, files in os.walk(root_path) if files]
    all_files = [item for sublist in f for item in sublist]
    all_files = [x for x in all_files if not any([y in x for y in names_to_exclude])]
    all_files = [x for x in all_files if any([x.endswith(y) for y in ext_include])]
    return all_files
def write_list_to_file(file_name, L):
    f = open(file_name, 'w+', encoding='utf-8')
    f.write('\n'.join(L))
    f.close()
def read_file_to_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as f: # Accounts for encoded characters
        L = f.read().splitlines() # list of lines
    return L

root_path = "."
names_to_exclude = ["cython", "ConvertVelocityToCenter"]
ext_include = [".py"]
all_files = get_all_files_in_path_recursive(root_path, names_to_exclude, ext_include)
print('len(all_files) = '+str(len(all_files)))
print('\n'.join(all_files))

def replace_swap(s, a, b):
  s = s.replace(a, "UNIQUEHASHA0")
  s = s.replace(b, "UNIQUEHASHA1")
  s = s.replace("UNIQUEHASHA0", b)
  s = s.replace("UNIQUEHASHA1", a)
  return s

for f in all_files:
  L = read_file_to_list(f)
  s = '\n'.join(L)
  for i in ("i", "i_env", "i_gm", "j"):
    for q in ("q", "q_new", "q_tendencies"):
      s_old = q+"['w', "+i+"][k]"
      s_new = q+"['w', "+i+"].Mid(k)"
      s = replace_swap(s, s_old, s_new)

      s_old = q+"['w', "+i+"].Identity(k)"
      s_new = q+"['w', "+i+"].Mid(k)"
      s = replace_swap(s, s_old, s_new)

      s_old = q+"['w', "+i+"].Cut(k)"
      s_new = q+"['w', "+i+"].DualCut(k)"
      s = replace_swap(s, s_old, s_new)

      s_old = q+"['w', "+i+"][slice_real_n]"
      s_new = q+"['w', "+i+"][slice_real_c]"
      s = replace_swap(s, s_old, s_new)

      s = s.replace(q+"['w', "+i+"].Identity(k)", q+"['w', "+i+"][k]")

  L = s.split('\n')
  write_list_to_file(f, L)



