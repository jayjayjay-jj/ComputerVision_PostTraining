print('Hello')


# Assign variable
name = 'Jayyy'
words = "ababc xxx"
print('%s said %s'%(name, words))
print('{} said {}'.format(name, words))
print('{1} said {0}'.format(name, words))
print(f'{name} said {words}')

phi = 3.1428
print('{:.2f}'.format(phi))  # Only print until 2 decimal
print('%.2f'%(phi))


print('')

# List
print('List')

my_list = []
print(type(my_list))

    # Append List
my_list.append([1, 2, 3])
my_list.append([4, 5, 6])
print(my_list)

for i in my_list:
    for j in i:
        print(j)


print('')

# Tuple -> unchangeable
print('Tuple')

my_tuple = ('a', 'b', 'c')
x, y, z = my_tuple

print(my_tuple)
print(x, y, z)

for i in my_tuple:
    for j in i:
        print(j)


print('')

# Set -> unordered (random printing), can't be duplicated while inserting
print('Set')

my_set = {'a', 'b', 'a', 3, 3, 3}
print(my_set)


print('')

# Dictionary
print('Dictionary')

my_dict = {
    'key1' : 2,
    'key2' : 5,
    'key3': 500
}
print(my_dict['key2'])
print(my_dict.items())