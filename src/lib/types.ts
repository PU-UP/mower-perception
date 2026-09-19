export type MowerClass = {
  id: number
  name: string
  name_zh: string
  color: number[]
  traversable: boolean
  safety: boolean
}

export type ClassStat = MowerClass & {
  pixels: number
  ratio: number
}

export type InferResponse = {
  overlay: string
  mask: string
  input: string
  stats: {
    classes: ClassStat[]
    traversable_ratio: number
    safety_ratio: number
    pixels: number
    latency_ms: number
    model: string
    device: string
  }
  taxonomy: MowerClass[]
}

export type SampleMeta = {
  id: string
  file: string
  title: string
  note: string
}

export type TaxonomyResponse = {
  model: string
  classes: MowerClass[]
  samples: SampleMeta[]
}
