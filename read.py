def read(file):
	count=0
	with open(file, 'r') as reader:
		# Create the tensor
		tns = alto_t()

		x=0
		for row in reader:
			#print(count)
			count=count+1
			#if count % 1000 == 0:
				#print("Count: ", count, "Max Chain Depth:", tns.max_chain_depth)
			row = row.split()
			# Get the value
			val = float(row.pop())
			# The rest of the line is the indexes
			idx = [int(i) for i in row]
			x=x+1

			tns.set(idx, val)

	reader.close()
	return tns

def write(file, tns):
	for bucket in tns.table:
		if bucket == None:
			continue
		for item in bucket:
			print(*mort.decode(item[0], tns.nmodes), item[1])

