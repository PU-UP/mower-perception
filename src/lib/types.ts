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

export type ModelCard = {
  id: string
  display_name: string
  task: string
  loader: string
  hub_id: string | null
  output_taxonomy: string
  requires_remapping: boolean
  num_classes: number | null
  input_long_side: number
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
    model_id?: string
    device: string
    backend?: string
    task?: string
  }
  model?: {
    id: string
    display_name: string
    hub_id?: string | null
    backend: string
    output_taxonomy?: string
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
  model_id?: string
  display_name?: string
  backend?: string
  task?: string
  device?: string
  output_taxonomy?: string
  default_model?: string
  models?: ModelCard[]
  classes: MowerClass[]
  samples: SampleMeta[]
}
