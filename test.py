def allLongestStrings(inputArray):
    lenn = 0
    out = []
    for i in inputArray:
        if len(i) > lenn:
            lenn = len(i)
    print(lenn)
    for i in inputArray:
        print(i + "   " +  str(len(i)))
        if len(i) == lenn:
            out.append(i)
    return out

a = ["a",
 "abc",
 "cbd",
 "zzzzzz",
 "a",
 "abcdef",
 "asasa",
 "aaaaaa"]
print(allLongestStrings(a))
# print(checkPalindrome("aba"))