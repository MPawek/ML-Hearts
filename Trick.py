from Card import Card, Suit

hearts = 3 # the corresponding index to the suit hearts
spades = 2
queen = 36

class Trick:
	def __init__(self):
		self.trick = [-1, -1, -1, -1]
		self.suit = -1
		self.cardsInTrick = 0
		self.points = 0
		self.highest = 0 # rank of the high trump suit card in hand
		self.winner = -1

	def reset(self):
		self.trick = [-1, -1, -1, -1]
		self.suit = -1
		self.cardsInTrick = 0
		self.points = 0
		self.highest = 0
		self.winner = -1

	# def cardsInTrick(self):
	# 	count = 0
	# 	for card in self.trick:
	# 		if card is not 0:
	# 			count += 1
	# 	return count

	def setTrickSuit(self, card):
		self.suit = card // 13

	def addCard(self, card, index):
		if self.cardsInTrick == 0: # if this is the first card added, set the trick suit
			self.setTrickSuit(card)
			# print('Current trick suit: ', self.suit)

		self.trick[index] = card
		self.cardsInTrick += 1

		if card // 13 == hearts:
			self.points += 1
		elif card == queen:
			self.points += 13


		# If card is the same suit as trick suit:
		if card // 13 == self.suit:
			if card > self.highest:
				self.highest = card
				self.winner = index
				# print( "Highest: ", self.highest)
