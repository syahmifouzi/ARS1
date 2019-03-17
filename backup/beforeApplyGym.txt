package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

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

type rollouts struct {
	rPos, rNeg float64
	d          [][]float64
}

type env struct {
	nbInputs, nbOutputs int
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

func (p *Policy) update(rollout []rollouts, sigmaR float64, hp Hp) {

	step := zeros(len(rollout[0].d), len(rollout[0].d[0]))

	for _, v := range rollout {
		s1 := v.rPos - v.rNeg
		s2 := darabN(v.d, s1)

		step = tambah(step, s2)
	}

	ss1 := float64(hp.nbBestDirections) * sigmaR
	ss1 = float64(hp.learningRate) / ss1
	ss2 := darabN(step, ss1)

	(*p).theta = tambah((*p).theta, ss2)
}

// RESET = 1x26
// STATE = 1x26 (input)
// ACTION = 1x6 (output)

func explore(hp Hp, normalizer Normalizer, policy Policy, direction string, delta [][]float64) float64 {
	// I assume reset the robot leg to default
	// state := env.reset()
	var state [][]float64

	done := false
	var numPlays, sumRewards, reward float64

	// Calculate the accumulate reward on the full episode
	for !done && numPlays < float64(hp.episodeLength) {
		normalizer.observe(state)
		state = normalizer.normalize(state)
		// action := policy.evaluate(state, delta, direction, hp)

		// state, reward, done = env.step(action)

		if reward < -1 {
			reward = -1
		} else if reward > 1 {
			reward = 1
		}
		sumRewards += reward
		numPlays++
	}

	return sumRewards
}

func train(hp Hp, p Policy, normalizer Normalizer, inputSize, outputSize int) {
	for step := 0; step < hp.nbSteps; step++ {
		// Initializing the pertubation deltas and the positive/negative rewards
		deltas := p.sampleDeltas(inputSize, outputSize, hp)
		positiveRewards := zeros(1, hp.nbDirections)
		negativeRewards := zeros(1, hp.nbDirections)

		// Getting the positive rewards in the positive directions
		for k := 0; k < hp.nbDirections; k++ {
			positiveRewards[0][k] = explore(hp, normalizer, p, "positive", deltas[k])
		}

		// Getting the negative rewards in the negative/positive directions
		for k := 0; k < hp.nbDirections; k++ {
			negativeRewards[0][k] = explore(hp, normalizer, p, "negative", deltas[k])
		}

		// Gathering all the positive/negative rewards to compute the standard deviation of these rewards
		// Concat both into 1 array
		allRewards := concatArr(positiveRewards, negativeRewards)
		// go lib for std might be a bit different than python
		sigmaR := std(allRewards)

		// Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
		order := getOrder(positiveRewards, negativeRewards)
		rollout := make([]rollouts, len(positiveRewards[0]))
		for i, v := range order {
			rollout[i].rPos = positiveRewards[0][v]
			rollout[i].rNeg = negativeRewards[0][v]
			rollout[i].d = deltas[v]
		}

		// Updating the policy
		p.update(rollout, sigmaR, hp)

		// Printing the final reward of the policy after the update
		rewardEvaluation := explore(hp, normalizer, p, "none", deltas[0])
		fmt.Println("Step:", step, "Rewards:", rewardEvaluation)
	}
}

func main() {
	fmt.Println("Start")

	hp := Hp{}
	hp.init()

	// Number of input and output depend on our own gym
	nbInputs := 26
	nbOutputs := 6

	policy := Policy{}
	policy.init(nbInputs, nbOutputs)

	normalizer := Normalizer{}
	normalizer.init(nbInputs)

	train(hp, policy, normalizer, nbInputs, nbOutputs)

	fmt.Println("End")
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

func std(m1 [][]float64) float64 {
	// https://www.gonum.org/post/intro-to-stats-with-gonum/
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

func getOrder(m1 [][]float64, m2 [][]float64) []int {

	m := make(map[float64]int)

	s := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		s[i] = make([]float64, len(m1[0]))
	}

	k := 0
	for i, v := range m1 {
		for i2, v2 := range v {
			if v2 > m2[i][i2] {
				s[i][i2] = v2
				m[v2] = k
				k++
			} else {
				s[i][i2] = m2[i][i2]
				m[m2[i][i2]] = k
				k++
			}
		}
	}
	sort.Float64s(s[0])

	var r []int

	for _, val := range s[0] {
		r = append(r, m[val])
		// fmt.Println(m[val], val)
	}
	// fmt.Println(r)
	return r
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
