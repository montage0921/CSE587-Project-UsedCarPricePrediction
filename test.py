shadow_info = "2 OwnersNo flood or frame damageNo odometer problems"
new_shadow_info = ""  # Create a new string to store the result

for i in shadow_info:
    if i == 'N':
        new_shadow_info += ','  # Insert a comma before 'N'
    new_shadow_info += i  # Add the current character to the new string

print(new_shadow_info)
