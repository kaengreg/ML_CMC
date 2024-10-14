from typing import List

def hello(name: str = None) -> str:
    if not name:
        return 'Hello!'
    else:
        return f'Hello, {name}!'



def int_to_roman(num: int) -> str:
    nums = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    symbols = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC', 'C', 'CD', 'D', 'CM', 'M']

    res = ''
    for i in range(len(symbols) - 1, -1, -1):
        res += num // nums[i] * symbols[i]
        num %= nums[i]

    return res

def longest_common_prefix(strs_input: List[str]) -> str:
    res = ''
    strs = [i.strip() for i in strs_input]
    print(strs_input)
    print(strs)
    if len(strs):
        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != strs[0][i]:
                    return res
            res += strs[0][i]

    return res


def primes() -> int:
    def isPrime(n):
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False

        return True

    i = 2
    while True:
        if isPrime(i):
            yield i
        i += 1

class BankCard:
    def __init__(self, total_sum, balance_limit = -1):
        if total_sum < 0:
            raise ValueError("Total sum must be a non-negative number.")

        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")

        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")

    def __str__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        if not isinstance(other, BankCard):
            raise TypeError("Can only merge with another BankCard.")

        total_sum = self.total_sum + other.total_sum
        balance_limit = max(self.balance_limit,
                            other.balance_limit) if self.balance_limit != 0 or other.balance_limit != 0 else None

        return BankCard(total_sum, balance_limit)

    @property
    def balance(self):
        if self.balance_limit != 0:
            self.balance_limit -= 1
            return self.total_sum
        else:
            raise ValueError("Balance check limits exceeded.")

    def put(self, sum_put):
        if sum_put < 0:
            raise ValueError("Amount to put must be a non-negative number.")

        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")
