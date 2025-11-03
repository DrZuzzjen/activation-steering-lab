<svelte:options accessors={true} />

<script lang="ts">
	import type { Gradio } from "@gradio/utils";
	import { Block, BlockTitle } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import type { LoadingStatus } from "@gradio/statustracker";
	import { onMount, onDestroy } from "svelte";
	import { createBrainScene } from "./BrainScene";

	export let gradio: Gradio<{
		change: never;
		submit: never;
		input: never;
		clear_status: LoadingStatus;
	}>;
	export let label = "3D Brain Activation Viewer";
	export let elem_id = "";
	export let elem_classes: string[] = [];
	export let visible = true;
	export let value: any = null;
	export let show_label: boolean;
	export let scale: number | null = null;
	export let min_width: number | undefined = undefined;
	export let loading_status: LoadingStatus | undefined = undefined;
	export let interactive: boolean = true;

	let canvas: HTMLCanvasElement;
	let scene: any = null;
	let container: HTMLDivElement;

	onMount(() => {
		if (canvas && value) {
			try {
				scene = createBrainScene(canvas, value);
			} catch (error) {
				console.error("Error creating brain scene:", error);
			}
		}
	});

	onDestroy(() => {
		if (scene && scene.dispose) {
			scene.dispose();
		}
	});

	// Update scene when value changes
	$: if (scene && value) {
		try {
			scene.updateActivations(value);
		} catch (error) {
			console.error("Error updating activations:", error);
		}
	}

	// When the value changes, dispatch the change event
	$: value, gradio.dispatch("change");
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={true}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<div class="brain-viewer-container" bind:this={container}>
		{#if show_label}
			<BlockTitle {show_label} info={undefined}>{label}</BlockTitle>
		{/if}
		<canvas bind:this={canvas} class="brain-canvas"></canvas>
		{#if value && value.metadata}
			<div class="info-panel">
				<div class="info-item">
					<strong>Concept:</strong> {value.metadata.concept_name}
				</div>
				<div class="info-item">
					<strong>Injection Layer:</strong> {value.metadata.injection_layer}
				</div>
				<div class="info-item">
					<strong>Peak Layer:</strong> {value.peak_activation_layer}
				</div>
			</div>
		{/if}
	</div>
</Block>

<style>
	.brain-viewer-container {
		width: 100%;
		height: 800px;
		background: rgb(10, 10, 10);
		border-radius: 8px;
		overflow: hidden;
		position: relative;
	}

	.brain-canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.info-panel {
		position: absolute;
		top: 10px;
		left: 10px;
		background: rgba(0, 0, 0, 0.7);
		color: white;
		padding: 10px;
		border-radius: 4px;
		font-size: 12px;
		pointer-events: none;
	}

	.info-item {
		margin: 4px 0;
	}
</style>
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={true}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<label class:container>
		<BlockTitle {show_label} info={undefined}>{label}</BlockTitle>

		<input
			data-testid="textbox"
			type="text"
			class="scroll-hide"
			bind:value
			bind:this={el}
			{placeholder}
			disabled={!interactive}
			dir={rtl ? "rtl" : "ltr"}
			on:keypress={handle_keypress}
		/>
	</label>
</Block>

<style>
	label {
		display: block;
		width: 100%;
	}

	input {
		display: block;
		position: relative;
		outline: none !important;
		box-shadow: var(--input-shadow);
		background: var(--input-background-fill);
		padding: var(--input-padding);
		width: 100%;
		color: var(--body-text-color);
		font-weight: var(--input-text-weight);
		font-size: var(--input-text-size);
		line-height: var(--line-sm);
		border: none;
	}
	.container > input {
		border: var(--input-border-width) solid var(--input-border-color);
		border-radius: var(--input-radius);
	}
	input:disabled {
		-webkit-text-fill-color: var(--body-text-color);
		-webkit-opacity: 1;
		opacity: 1;
	}

	input:focus {
		box-shadow: var(--input-shadow-focus);
		border-color: var(--input-border-color-focus);
	}

	input::placeholder {
		color: var(--input-placeholder-color);
	}
</style>
