#!/usr/bin/env python3
import torch

torch.manual_seed(42)
# get sum
def sum():
    a = torch.tril(torch.ones(3, 3))
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b

    print('a = ')
    print(a)
    print('---')
    print('b = ')
    print(b)
    print('---')
    print('c = ')
    print(c)
    print('---')
    
def get_average():
    a = torch.tril(torch.ones(3, 3))
    a = a / torch.sum(a, dim=1, keepdim=True)
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b

    print('a = ')
    print(a)
    print('---')
    print('b = ')
    print(b)
    print('---')
    print('c = ')
    print(c)
    print('---')

def main():
    get_average()


if __name__ == '__main__':
    main()
