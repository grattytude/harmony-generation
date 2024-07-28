# genetic/genetic.py
import random
from dataclasses import dataclass
import music21

@dataclass(frozen=True)
class MelodyData:
    notes: list
    duration: int = None  # Computed attribute
    number_of_bars: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(self, "duration", sum(duration for _, duration in self.notes))
        object.__setattr__(self, "number_of_bars", int(self.duration // 4))

class GeneticMelodyHarmonizer:
    def __init__(self, melody_data, chords, population_size, mutation_rate, fitness_evaluator):
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = self.fitness_evaluator.get_chord_sequence_with_highest_fitness(self._population)
        return best_chord_sequence

    def _initialise_population(self):
        return [self._generate_random_chord_sequence() for _ in range(self.population_size)]

    def _generate_random_chord_sequence(self):
        return [random.choice(self.chords) for _ in range(self.melody_data.number_of_bars)]

    def _select_parents(self):
        fitness_values = [self.fitness_evaluator.evaluate(seq) for seq in self._population]
        return random.choices(self._population, weights=fitness_values, k=self.population_size)

    def _create_new_population(self, parents):
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(parents[i], parents[i + 1]), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence):
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence

class FitnessEvaluator:
    def __init__(self, melody_data, chord_mappings, weights, preferred_transitions):
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        return sum(self.weights[func] * getattr(self, f"_{func}")(chord_sequence) for func in self.weights)

    def _chord_melody_congruence(self, chord_sequence):
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 4 and melody_index < len(self.melody_data.notes):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence):
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        score = 0
        if chord_sequence[0] in ["C", "Am"]:
            score += 1
        if chord_sequence[-1] in ["C"]:
            score += 1
        if "F" in chord_sequence and "G" in chord_sequence:
            score += 1
        return score / 3

def create_score(melody, chord_sequence, chord_mappings):
    score = music21.stream.Score()
    melody_part = music21.stream.Part()
    for note_name, duration in melody:
        melody_note = music21.note.Note(note_name, quarterLength=duration)
        melody_part.append(melody_note)
    chord_part = music21.stream.Part()
    current_duration = 0
    for chord_name in chord_sequence:
        chord_notes_list = chord_mappings.get(chord_name, [])
        chord_notes = music21.chord.Chord(chord_notes_list, quarterLength=4)
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += 4
    score.append(melody_part)
    score.append(chord_part)
    return score

def midi_to_melody(midi_path):
    midi_data = music21.converter.parse(midi_path)
    melody = []
    for element in midi_data.flatten().notes:
        if isinstance(element, music21.note.Note):
            melody.append((element.nameWithOctave, element.quarterLength))
    return melody

def detect_key(midi_path):
    midi_data = music21.converter.parse(midi_path)
    key = midi_data.analyze('key')
    return key

def generate_chord_mappings_and_transitions(key):
    tonic_name = key.tonic.name
    scale = music21.scale.MajorScale(tonic_name) if key.mode == 'major' else music21.scale.MinorScale(tonic_name)
    
    chord_mappings = {
        tonic_name: [str(scale.pitchFromDegree(1)), str(scale.pitchFromDegree(3)), str(scale.pitchFromDegree(5))],
        str(music21.interval.Interval('P4').transposePitch(music21.pitch.Pitch(tonic_name))): [str(scale.pitchFromDegree(2)), str(scale.pitchFromDegree(4)), str(scale.pitchFromDegree(6))],
        str(music21.interval.Interval('P5').transposePitch(music21.pitch.Pitch(tonic_name))): [str(scale.pitchFromDegree(3)), str(scale.pitchFromDegree(5)), str(scale.pitchFromDegree(7))],
    }
    
    # Adjust preferred transitions to include all generated chords
    preferred_transitions = {
        tonic_name: [
            music21.interval.Interval('P5').transposePitch(music21.pitch.Pitch(tonic_name)),
            music21.interval.Interval('M3').transposePitch(music21.pitch.Pitch(tonic_name)),
            music21.interval.Interval('P4').transposePitch(music21.pitch.Pitch(tonic_name))
        ],
        str(music21.interval.Interval('P4').transposePitch(music21.pitch.Pitch(tonic_name))): [
            music21.interval.Interval('P5').transposePitch(music21.pitch.Pitch(tonic_name)),
            music21.interval.Interval('M3').transposePitch(music21.pitch.Pitch(tonic_name))
        ],
        str(music21.interval.Interval('M3').transposePitch(music21.pitch.Pitch(tonic_name))): [
            music21.interval.Interval('P4').transposePitch(music21.pitch.Pitch(tonic_name)),
            music21.interval.Interval('P5').transposePitch(music21.pitch.Pitch(tonic_name))
        ],
    }
    
    # Ensure all chords are covered in preferred transitions
    all_chords = list(chord_mappings.keys())
    for chord in all_chords:
        if chord not in preferred_transitions:
            preferred_transitions[chord] = []
    
    return chord_mappings, preferred_transitions

def main(input_midi_path, output_midi_path):
    melody = midi_to_melody(input_midi_path)
    key = detect_key(input_midi_path)
    print(f"Detected Key: {key}")

    weights = {
        "chord_melody_congruence": 0.4,
        "chord_variety": 0.1,
        "harmonic_flow": 0.3,
        "functional_harmony": 0.2
    }

    chord_mappings, preferred_transitions = generate_chord_mappings_and_transitions(key)

    melody_data = MelodyData(melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    generated_chords = harmonizer.generate(generations=1000)

    music21_score = create_score(melody, generated_chords, chord_mappings)
    music21_score.write('midi', fp=output_midi_path)
    print(f"MIDI file created at: {output_midi_path}")

if __name__ == "__main__":
    input_midi_path = "gool.mid"
    output_midi_path = "final14362.mid"
    main(input_midi_path, output_midi_path)
