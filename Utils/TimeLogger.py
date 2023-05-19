import datetime

logmsg = ''
timemark = dict()
saveDefault = False
def log(msg, save=None, oneline=False):
	"""Add msg to logmsg and print msg if needed.

	Args:
		msg: Messages to be printed.
		save: A flag to decide whether to add msg to logmsg.
		oneline: A flag to decide whether print msg.

	Returns:
		None
	"""
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

if __name__ == '__main__':
	log('')