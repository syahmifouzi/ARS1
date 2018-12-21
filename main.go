package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat"
)

// Hp is Hyper Parameter
type Hp struct {
	nbSteps, episodeLength, nbDirections, nbBestDirections int
	learningRate, noise                                    float64
}

// Normalizer is Normalizer
type Normalizer struct {
	n, mean, meanDiff, variance [][]float64
}

// Policy is the AI
type Policy struct {
	theta [][]float64
}

func (hp *Hp) init() {
	(*hp).nbSteps = 3
	(*hp).episodeLength = 1000
	(*hp).nbDirections = 16
	(*hp).nbBestDirections = 16
	(*hp).learningRate = 0.02
	(*hp).noise = 0.03
}

func (nm *Normalizer) init(nbInputs int) {
	(*nm).n = zeros(1, nbInputs)
	(*nm).mean = zeros(1, nbInputs)
	(*nm).meanDiff = zeros(1, nbInputs)
	(*nm).variance = zeros(1, nbInputs)
}

func (nm *Normalizer) observe(x [][]float64) {
	(*nm).n = tambahN((*nm).n, 1)
	lastMean := (*nm).mean
	mean1 := tolak(x, (*nm).mean)
	mean1 = bahagi(mean1, (*nm).n)
	(*nm).mean = tambah((*nm).mean, mean1)
	meanDiff1 := tolak(x, lastMean)
	meanDiff1 = darab(meanDiff1, tolak(x, (*nm).mean))
	(*nm).meanDiff = tambah((*nm).meanDiff, meanDiff1)
	variance1 := bahagi((*nm).meanDiff, (*nm).n)
	variance1 = clipMin(variance1, 1e-2)
	(*nm).variance = variance1
}

func (nm *Normalizer) normalize(inputs [][]float64) [][]float64 {
	obsMean := (*nm).mean
	obsStd := sqrt((*nm).variance)
	r1 := tolak(inputs, obsMean)
	r1 = bahagi(r1, obsStd)
	return r1
}

func (p *Policy) init(inputSize, outputSize int) {
	(*p).theta = zeros(outputSize, inputSize)
}

func (p *Policy) evaluate(input, delta [][]float64, direction string, hp Hp) [][]float64 {
	switch direction {
	case "none":
		return dot(input, (*p).theta)
	case "positive":
		r := darabN(delta, hp.noise)
		r = tambah((*p).theta, r)
		return dot(input, transpose(r))
	default:
		r := darabN(delta, hp.noise)
		r = tolak((*p).theta, r)
		return dot(input, transpose(r))
	}
}

func (p *Policy) sampleDeltas(inputSize, outputSize int, hp Hp) [][][]float64 {
	var r [][][]float64
	for i := 0; i < hp.nbDirections; i++ {
		r = append(r, randomizeValue(outputSize, inputSize))
	}
	return r
}

func (p *Policy) update(inputSize, outputSize int) {

}

func train(hp Hp, p Policy) {
	for step := 0; step < hp.nbSteps; step++ {
		// Initializing the pertubation deltas and the positive/negative rewards
		// deltas := p.sampleDeltas(nbInputs, nbOutputs, hp)
		// positiveRewards := zeros(1, hp.nbDirections)
		// negativeRewards := zeros(1, hp.nbDirections)

		// Getting the positive rewards in the positive directions
		for k := 0; k < hp.nbDirections; k++ {
			// positiveRewards[0][k] = explore()
		}

		// Getting the negative rewards in the positive directions
		for k := 0; k < hp.nbDirections; k++ {
			// negativeRewards[0][k] = explore()
		}

		// Gathering all the positive/negative rewards to compute the standard deviation of these rewards
		// Concat both into 1 array
		// allRewards := concatArr(positiveRewards, negativeRewards)
	}
}

func main() {

	// hp := Hp{3, 1000, 16, 16, 0.02, 0.03}
	hp := Hp{}
	hp.init()
	var nbInputs = 26
	var nbOutputs = 6
	normalizer := Normalizer{}
	normalizer.init(nbInputs)

	var input = [][]float64{{-0.27056041, 0., 0., -0.20892696, 0., 0.17776233,
		-0., 0.31310795, 0.35891473, -0.06750051, -0.4259459, 0.01321563,
		0.45592347, -0.08710772, 0.27216179, -0.09448055, 0.32598927, -0.07648785,
		-0.25519671, 0.17547957, 0.33333333, 0.46534397, 0.41015156, 0.12751534,
		-0.15681251, 0.34433738}}

	var delta = [][]float64{{-0.80217284, -0.44887781, -1.10593508, -1.65451545, -2.3634686, 1.13534535,
		-1.01701414, 0.63736181, -0.85990661, 1.77260763, -1.11036305, 0.18121427,
		0.56434487, -0.56651023, 0.7299756, 0.37299379, 0.53381091, -0.0919733,
		1.91382039, 0.33079713, 1.14194252, -1.12959516, -0.85005238, 0.96082,
		-0.21741818, 0.15851488},
		{0.87341823, -0.11138337, -1.03803876, -1.00947983, -1.05825656, 0.65628408,
			-0.06249159, -1.73865429, 0.103163, -0.62166685, 0.27571804, -1.09067489,
			-0.60998525, 0.30641238, 1.69182613, -0.74795374, -0.58079722, -0.11075397,
			2.04202875, 0.44752069, 0.68338423, 0.02288597, 0.85723427, 0.18393058,
			-0.41611158, 1.25005005},
		{1.24829979, -0.75767414, 0.58829416, 0.34685933, 1.3670327, 0.67371607,
			-1.2915627, -0.84824392, -0.16659957, 0.91719602, 0.08025059, 0.22823877,
			-0.8804768, 0.27812885, -0.07015677, 0.62958793, -1.81342356, 1.54744858,
			0.32505743, -0.21191292, -1.54672407, 1.04520063, 1.01037548, 0.07083664,
			0.71758983, -0.25070491},
		{-0.05152993, 0.01312891, 0.20223906, 0.45495224, -0.39926817, 0.18106742,
			0.80748795, 0.81253519, 0.21090203, 0.42177915, 0.58192518, -0.41020752,
			2.2968661, 1.68849705, 0.62581147, -1.61136381, 0.06009774, 0.46242079,
			0.68483649, -0.59546033, 0.99905124, -0.30817074, 0.36583834, 1.60750704,
			-0.23817737, -0.34082828},
		{0.48759421, 1.73907303, 0.0689698, 0.47324139, -0.65035502, -0.77910696,
			-0.77766271, 0.6225628, 0.42756207, 0.0740096, -0.4531686, 0.60415364,
			2.38520581, -0.12388333, -0.32419367, 0.31075423, 2.46162831, -0.31612369,
			-1.81506277, 0.6842495, 0.03203253, 0.19627021, 0.90745116, -2.13483482,
			0.81684718, 1.18417131},
		{-0.20448056, -0.11084446, 1.41448273, -1.416645, 0.67351346, -0.77229442,
			-0.09387704, -0.16977402, -0.54114463, 0.53794761, 0.39128265, 2.21191487,
			-0.16224463, 0.29117816, 0.10806266, -0.19953292, 0.2328323, 0.15539326,
			0.59372515, -1.35055772, 0.83056467, 0.11321804, -1.24274572, 1.59948307,
			2.47441941, -0.33232485}}

	direction := "positive"

	p := Policy{}
	p.init(nbInputs, nbOutputs)
	r := p.evaluate(input, delta, direction, hp)
	fmt.Println("action: ", r)
	train(hp, p)

	var mm1 = [][]float64{{-966.80775636, -944.54456444, -941.39167427, -908.57375676, -942.50477587,
		-951.3645666, -909.90726117, -907.80823, -971.58509113, -956.02367439,
		-961.6827609, -445.07480596, -961.58943846, -963.00084701, -969.89787205,
		-965.95142945, -959.65110938, -875.6368449, -938.54339511, -954.19116807,
		-958.78896575, -945.44521011, -939.4083488, -954.22280762, -974.53632688,
		-955.28894524, -946.28926372, -949.71335999, -957.85133814, -899.48257535,
		-959.12524539, -924.69444581}}

	fmt.Println("std: ", std(mm1))

}

func randomizeValue(r int, c int) [][]float64 {
	// we are seeding the rand variable with present time
	// or use crypto/rand for more secure way
	// https://gobyexample.com/random-numbers
	// so that we would get different output each time
	// rand.Seed(time.Now().UnixNano())
	// OR WE CAN JUST CONSTANT IT FOR NOW!
	rand.Seed(0)

	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 2*rand.Float64() - 1
		}
	}

	return v
}

// func initWeight() {
// 	wa = randomizeValue(1, 1)
// 	ua = randomizeValue(1, 1)
// 	ba = zeros(1, 5)

// 	// get the dimensions of matrix
// 	// init neural network weights
// 	wf = randomizeValue(1, 1)
// 	uf = randomizeValue(1, 1)
// 	bf = zeros(1, 5)

// 	wi = randomizeValue(1, 1)
// 	ui = randomizeValue(1, 1)
// 	bi = zeros(1, 5)

// 	wo = randomizeValue(1, 1)
// 	uo = randomizeValue(1, 1)
// 	bo = zeros(1, 5)
// }

func zeros(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 0
		}
	}

	return v
}

func zerosAlpha(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = alpha
		}
	}

	return v
}

func std(m1 [][]float64) float64 {
	// var sum, mean, sd, d float64
	// for _, v := range m1[0] {
	// 	fmt.Println("value of v:", v)
	// 	sum += v
	// 	d++
	// }
	// mean = sum / d
	// for _, v := range m1[0] {
	// 	sd += math.Pow(v-mean, 2)
	// }

	// mean := stat.Mean(m1[0], nil)
	variance := stat.Variance(m1[0], nil)
	stddev := math.Sqrt(variance)

	return stddev
}

func concatArr(m1 [][]float64, m2 [][]float64) [][]float64 {
	c := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		c[i] = make([]float64, len(m1[0])+len(m2[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			c[i][i2] = v2
		}
	}

	for i, v := range m2 {
		for i2, v2 := range v {
			c[i][len(m1[0])+i2] = v2
		}
	}

	// r = append(r, randomizeValue(outputSize, inputSize))

	return c
}

func sigmoidDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = v2 * (1 - v2)
		}
	}

	return output
}

func sigmoid(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			var nX float64
			nX = 0 - v2
			output[i][i2] = 1 / (1 + math.Exp(nX))
		}
	}

	return output
}

func tanH(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = math.Tanh(v2)
		}
	}

	return output
}

func tanHDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(math.Tanh(v2), 2)
		}
	}

	return output
}

func oneMinusSquare(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(v2, 2)
		}
	}

	return output
}

func dot(m1 [][]float64, m2 [][]float64) [][]float64 {

	// Ref 2d slice
	// https://gobyexample.com/slices
	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m2[0]))
	}

	// outR := len(m1)
	// outC := len(m2[0])

	for outR, v := range m1 {
		for outC := range m2[0] {
			output[outR][outC] = 0
			for i2, v2 := range v {
				output[outR][outC] += v2 * m2[i2][outC]
			}
		}
	}

	return output
}

func darab(m1 [][]float64, m2 [][]float64) [][]float64 {

	product := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			product[i][i2] = v2 * m2[i][i2]
		}
	}

	return product
}

func darabN(m1 [][]float64, n float64) [][]float64 {

	product := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			product[i][i2] = v2 * n
		}
	}

	return product
}

func bahagi(m1 [][]float64, m2 [][]float64) [][]float64 {

	division := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		division[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			division[i][i2] = v2 / m2[i][i2]
		}
	}

	return division
}

func tambah(m1 [][]float64, m2 [][]float64) [][]float64 {

	sum := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		sum[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			sum[i][i2] = v2 + m2[i][i2]
		}
	}

	return sum
}

func tambahN(m1 [][]float64, n float64) [][]float64 {

	sum := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		sum[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			sum[i][i2] = v2 + n
		}
	}

	return sum
}

func tolak(m1 [][]float64, m2 [][]float64) [][]float64 {

	difference := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		difference[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			difference[i][i2] = v2 - m2[i][i2]
		}
	}

	return difference
}

func clipMin(m1 [][]float64, x float64) [][]float64 {

	clip := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		clip[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			if v2 < x {
				clip[i][i2] = x
			} else {
				clip[i][i2] = v2
			}
		}
	}

	return clip
}

func transpose(m1 [][]float64) [][]float64 {

	mT := make([][]float64, len(m1[0]))
	for i := 0; i < len(m1[0]); i++ {
		mT[i] = make([]float64, len(m1))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			mT[i2][i] = v2
		}
	}

	return mT
}

func sqrt(m1 [][]float64) [][]float64 {

	s := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		s[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			s[i][i2] = math.Sqrt(v2)
		}
	}

	return s
}

func softmax(m1 [][]float64) [][]float64 {

	sumZExp := 0.00
	s := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		s[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			s[i][i2] = math.Exp(v2)
		}
	}

	for _, v := range s {
		for _, v2 := range v {
			sumZExp += v2
		}
	}

	for i, v := range s {
		for i2, v2 := range v {
			s[i][i2] = v2 / sumZExp
		}
	}

	return s
}
