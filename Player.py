from Hand import Hand

class Player:
	def __init__(self, name, auto=False):
			self.name = name
			self.hand = Hand()
			self.score = 0
			self.roundScore = 0
			self.tricksWon = []

	def addCard(self, card):
		self.hand.addCard(card)


	def getInput(self, option):
		card = None
		while card is None:
			card = input(self.name + ", select a card to " + option + ": ")
		return card

	def play(self, option='play', c=None, auto=False):
		if auto:
			card = self.hand.getRandomCard()
		elif c is None:
			card = self.getInput(option)
		else:
			card = c
		if not auto:
			card = self.hand.playCard(card)
		return card

	def printHand(self):
		self.hand.__str__()

	def removeCard(self, card):
		self.hand.removeCard(card)
		if card == 0:
			self.hand.contains2ofclubs = False

	def trickWon(self, trick):
		self.roundScore += trick.points


	def hasSuit(self, suit):

		if suit == 0 and 1 in self.hand.hand[0:13]:
			return True
		elif suit == 1 and 1 in self.hand.hand[13:26]:
			return True
		elif suit == 2 and 1 in self.hand.hand[26:39]:
			return True
		elif suit == 3 and 1 in self.hand.hand[39:52]:
			return True

		return False

	def discardTricks(self):
		self.tricksWon = []

	def hasOnlyHearts(self):
		return self.hand.hasOnlyHearts()

	def startingHasOnlyHearts(self):
		return self.hand.startingHasOnlyHearts()