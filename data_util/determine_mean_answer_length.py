import statistics

total_lines = []

with open("total_answer_lengths.txt", "r") as file:
   for line in file:
       total_lines.append(int(line))

result = statistics.mean(total_lines)
print(f"Mean is {result:.2f}")